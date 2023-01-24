import os
import numpy as np
from mnist import MNIST
from tqdm import tqdm
import matplotlib.pyplot as plt
from fire import Fire
from sklearn.model_selection import train_test_split
import pandas as pd

try:
    from numba import njit
except ModuleNotFoundError:
    njit = lambda i: i


# @njit
# def vdsp(w, vmem, lr=1):
#     alpha = 1
#     vapp = -vmem
#     f_pot = np.exp(-alpha * w) * (-w + 1)
#     f_dep = np.exp(alpha * (w - 1)) * w
#     cond_pot = vapp > 0
#     cond_dep = vapp < 0
#     g_pot = np.exp(vapp) - 1
#     g_dep = np.exp(-vapp) - 1

#     dW = cond_pot * f_pot * g_pot - cond_dep * f_dep * g_dep
#     return dW * lr


@njit
def vdsp(w, vmem, lr=1):
    f_pot = 1 - w
    f_dep = w

    cond_pot = vmem < 0
    cond_dep = vmem > 0

    exp_vapp = np.exp(-vmem)

    g_pot = exp_vapp - 1
    g_dep = (1 / exp_vapp) - 1

    dW = (cond_pot * f_pot * g_pot - cond_dep * f_dep * g_dep) * lr
    return dW


@njit
def stdp(t_pre, t_post, lr=0.0005, tau_stdp=30):
    dW = np.sign(t_post - t_pre) * lr * np.exp(-np.abs(t_post - t_pre) / tau_stdp)
    return dW


@njit
def normalize_weights(weights, scale):
    post_sum = np.sum(weights, axis=0) + 1e-4
    weights = scale * weights / post_sum
    return weights


# @njit
def run_one_sample(
    X,
    mem_pot_input,
    mem_pot_output,
    weights,
    duration_per_sample,
    duration_between_samples,
    input_leak_cst,
    input_bias,
    input_threshold,
    input_reset,
    output_leak_cst,
    refractory_period,
    ### Lateral inhibition parameters
    lateral_inhibition_period,
    ### VDSP parameters
    use_vdsp,
    vdsp_lr,
    ### STDP parameters (if use_vdsp=false)
    dapre,
    dapost,
    tau_pre,
    tau_post,
    norm_scale,
    ### Adaptive threshold parameters (output layer)
    thresholds,
    th_inc,
    th_leak_cst,
    threshold_rest,
):
    refractory_neurons_input = np.zeros(mem_pot_input.shape[0])
    refractory_neurons = np.zeros(mem_pot_output.shape[0])
    recorded_input_spikes = np.zeros((mem_pot_input.shape[0], duration_per_sample))
    recorded_output_spikes = np.zeros((mem_pot_output.shape[0], duration_per_sample))
    if use_vdsp is False:
        last_spike_time_pre = np.zeros(mem_pot_input.shape[0])
        last_spike_time_post = np.zeros(mem_pot_output.shape[0])
    for t in range(duration_per_sample):
        thresholds = (thresholds - threshold_rest) * th_leak_cst + threshold_rest
        refractory_neurons = np.maximum(0.0, refractory_neurons - 1)
        refractory_neurons_input = np.maximum(0.0, refractory_neurons_input - 1)
        non_refrac_neurons = refractory_neurons == 0
        non_refrac_input_neurons = refractory_neurons_input == 0

        weight = 0.1
        X = weight*np.random.poisson(X/1000, size=784).clip(0, 1)



        mem_pot_input[non_refrac_input_neurons] = (
            mem_pot_input[non_refrac_input_neurons] * input_leak_cst + input_bias + X[non_refrac_input_neurons]
        )
        mem_pot_input[~non_refrac_input_neurons] = -1  # Refractory neurons are fixed at -1
        input_spikes = mem_pot_input > input_threshold
        mem_pot_input[input_spikes] = input_reset  # reset
        recorded_input_spikes[input_spikes, t] = 1
        refractory_neurons_input[input_spikes] = 2
        if use_vdsp is False:
            last_spike_time_pre[input_spikes] = t
            dw = stdp(t, last_spike_time_post, dapre, tau_pre)
            weights[input_spikes, :] += dw
            # with additive stdp, normalization is required
            weights = normalize_weights(weights, norm_scale)

        mem_pot_output[non_refrac_neurons] = (
            mem_pot_output[non_refrac_neurons] * output_leak_cst
            + input_spikes.astype(np.float64) @ weights[:, non_refrac_neurons]
        )
        output_spikes = mem_pot_output > thresholds
        if np.any(output_spikes):
            spiking_neuron = np.argmax(mem_pot_output - thresholds)  # This neuron is spiking
            thresholds[spiking_neuron] += th_inc
            if use_vdsp:
                dw = vdsp(weights[:, spiking_neuron], mem_pot_input, vdsp_lr)  # For its connection, get dws
                weights[:, spiking_neuron] += dw  # Apply plasticity
            else:
                last_spike_time_post[spiking_neuron] = t
                dw = stdp(last_spike_time_pre, t + 1, dapost, tau_post)
                weights[:, spiking_neuron] += dw  # Apply plasticity
                # with additive stdp, normalization is required
                weights = normalize_weights(weights, norm_scale)
            mem_pot_output[spiking_neuron] = -1  # reset mem pot
            refractory_neurons[spiking_neuron] = refractory_period  # Set refrac
            recorded_output_spikes[spiking_neuron, t] = 1

            # For all other output neurons, do lateral inhibition (clamp at 0 for lateral_inhibition_period)
            non_spiking_neurons = mem_pot_output != mem_pot_output[spiking_neuron]
            mem_pot_output[non_spiking_neurons] = 0
            refractory_neurons[non_spiking_neurons] = lateral_inhibition_period

    mem_pot_input = mem_pot_input * np.power(input_leak_cst, duration_between_samples)
    mem_pot_output = mem_pot_output * np.power(output_leak_cst, duration_between_samples)
    refractory_neurons = np.maximum(0.0, refractory_neurons - duration_between_samples)

    return mem_pot_input, mem_pot_output, weights, recorded_input_spikes, recorded_output_spikes

def save_dict_to_file(dic, filename):
    f = open(filename, "w")
    f.write(str(dic))
    f.close()



def main(
    seed=0x1B,
    n_output_neurons=10,
    duration_per_sample=350,
    duration_between_samples=100,
    input_leak_cst=np.exp(-1 / 30),
    output_leak_cst=np.exp(-1 / 60),
    input_threshold=1,
    output_threshold=1,
    input_reset=-1,
    input_bias=0.032,
    refractory_period=5,
    lateral_inhibition_period=10,
    input_scale=0.00675,
    nb_epochs=1,
    use_vdsp=True,
    vdsp_lr=0.001,
    tau_pre=150,
    tau_post=150,
    dapre=0.00000001,
    dapost=0.0001,
    norm_scale=600,
    weight_init_scale=1,
    with_validation=False,
    th_leak_cst=np.exp(-1 / 1000),
    th_inc=0.0,  # 0 is no adaptive threshold
    with_plots=True,
    normalize_duration=False,
):
    print(locals())

    argument_dict = locals()
    filename = f"accuracy_{seed}.csv"
    # save_dict_to_file(argument_dict, filename)

    df = pd.DataFrame.from_dict(argument_dict, orient="index")
    df = df.transpose()
    df.to_csv(filename, index=True)



    np.random.seed(seed)
    mndata = MNIST()
    images, labels = mndata.load_training()
    X_train, y_train = np.asarray(images), np.asarray(labels)
    X_train = X_train / 255 * input_scale

    if with_validation:
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33)
    else:
        images, labels = mndata.load_testing()
        X_test, y_test = np.asarray(images), np.asarray(labels)
        X_test = X_test / 255 * input_scale

    mem_pot_input = np.zeros(784)
    mem_pot_output = np.zeros(n_output_neurons)
    weights = np.random.uniform(0, weight_init_scale, size=(784, n_output_neurons))
    thresholds = np.ones(n_output_neurons) * output_threshold
    max_freq = 0

    spike_counts = np.zeros((10, n_output_neurons))  # For every class, keep track of spike count per neuron
    for epoch in range(nb_epochs):
        for i, (X, y) in enumerate(zip(tqdm(X_train), y_train)):
            mem_pot_input, mem_pot_output, weights, recorded_input_spikes, recorded_output_spikes = run_one_sample(
                X,
                mem_pot_input,
                mem_pot_output,
                weights,
                duration_per_sample,
                duration_between_samples,
                input_leak_cst,
                input_bias,
                input_threshold,
                input_reset,
                output_leak_cst,
                refractory_period,
                lateral_inhibition_period,
                use_vdsp=use_vdsp,
                vdsp_lr=vdsp_lr,
                tau_pre=tau_pre,
                tau_post=tau_post,
                dapre=dapre,
                dapost=dapost,
                norm_scale=norm_scale,
                thresholds=thresholds,
                th_inc=th_inc,
                th_leak_cst=th_leak_cst,
                threshold_rest=output_threshold,
            )
            if nb_epochs > 1 or i > len(y_train) // 2:
                spike_counts[y] += np.sum(recorded_output_spikes, axis=1)

            input_freqs = recorded_input_spikes.mean(axis=1) * 1000
            max_freq = np.maximum(max_freq, np.max(input_freqs))
            if normalize_duration:
                duration_per_sample = int(10 * 1000 / max_freq)

            if with_plots and (i + 1) % 5000 == 0:
                fig, axs = plt.subplots(2, n_output_neurons // 2)
                fig.suptitle(f"Receptive fields of output neurons at iteration {i+1}")
                axs = axs.flatten()
                for neuron in range(n_output_neurons):
                    if i > len(y_train) // 2:
                        axs[neuron].set_xlabel(f"{np.argmax(spike_counts, axis=0)[neuron]}")
                        axs[neuron].get_yaxis().set_visible(False)
                        axs[neuron].set_xticks([])
                    else:
                        axs[neuron].set_axis_off()
                    axs[neuron].imshow(weights[:, neuron].reshape(28, 28))  # , cmap="hot"

                plt.tight_layout()
                fig.savefig(f"mnist_wta_epoch_{epoch:02}_iteration_{i+1:05}.png", bbox_inches="tight")

    # Normalize the spike counts per neuron
    # sum_spike_counts = spike_counts.sum(axis=1)

    # Associate a label for every neuron based on its highest spike count per class
    labels = np.argmax(spike_counts, axis=0)
    nb_correct_classification_method_1 = 0
    nb_correct_classification_method_2 = 0
    for i, (X, y) in enumerate(zip(tqdm(X_test), y_test)):
        mem_pot_input, mem_pot_output, weights, recorded_input_spikes, recorded_output_spikes = run_one_sample(
            X,
            mem_pot_input,
            mem_pot_output,
            weights,
            duration_per_sample,
            duration_between_samples,
            input_leak_cst,
            input_bias,
            input_threshold,
            input_reset,
            output_leak_cst,
            refractory_period,
            lateral_inhibition_period,
            use_vdsp=use_vdsp,
            vdsp_lr=0,
            tau_pre=tau_pre,
            tau_post=tau_post,
            dapre=0,
            dapost=0,
            norm_scale=norm_scale,
            thresholds=thresholds,
            th_inc=0,
            th_leak_cst=1,  # Keep threshold as is
            threshold_rest=output_threshold,
        )

        ## Method 1: find the top spiking neuron and get its label
        sum_of_output_spikes = np.sum(recorded_output_spikes, axis=1)  # / sum_spike_counts
        top_spiking_neuron = np.argmax(sum_of_output_spikes)
        output_label = np.argmax(spike_counts[:, top_spiking_neuron])
        nb_correct_classification_method_1 += y == output_label

        ## Method 2: sum the spikes for all labeled neurons
        sum_of_spikes_for_sample = np.zeros(10)
        np.add.at(sum_of_spikes_for_sample, labels, sum_of_output_spikes)
        output_label = np.argmax(sum_of_spikes_for_sample)
        nb_correct_classification_method_2 += y == output_label

    print(f"Accuracy of method 1: {nb_correct_classification_method_1 / len(y_test):.5f}")
    print(f"Accuracy of method 2: {nb_correct_classification_method_2 / len(y_test):.5f}")
    print(f"Maximum spiking frequency of input layer: {max_freq}")
    return (
        max(nb_correct_classification_method_1 / len(y_test), nb_correct_classification_method_2 / len(y_test)),
        max_freq,
    )

def run_random_vdsp(seed):
    output_file = f"accuracy_{seed}.csv"
    np.random.seed(seed)
    accuracy = run_vdsp(
        seed=seed,
        n_output_neurons=10,
        duration_per_sample=350,
        duration_between_samples=100,
        input_leak_cst=np.exp(-1 / 30),
        output_leak_cst=np.exp(-1 / 60),
        # input_leak_cst=np.exp(-1 / np.random.uniform(30/2, 2*30)),
        # output_leak_cst=np.exp(-1 / np.random.uniform(60/2, 2*60)),
        input_threshold=1,
        output_threshold=10,
        input_reset=-1,
        # input_threshold=np.random.uniform(1/2, 2*1),
        # output_threshold=np.random.uniform(10/2, 2*10),
        # input_reset=np.random.uniform(-1/2, -2*1),
        input_bias=np.random.uniform(0.032/2, 2*0.032),
        input_scale=np.random.uniform(0.00675, 500),
        nb_epochs=1,
        use_vdsp=True,
        vdsp_lr=np.random.uniform(0.001/2, 5*0.001),   
        with_validation=False,
        th_leak_cst=np.exp(-1 / 1000),
        th_inc=0.0,  # 0 is no adaptive threshold
        # refractory_period=np.random.randint(1, 5),
        # lateral_inhibition_period=np.random.randint(1, 50),

    )[0]
    # argument_dict["accuracy"] = accuracy
    # save_dict_to_file(argument_dict, output_file)
    df = pd.read_csv(output_file)
    df["accuracy"] = accuracy
    df.to_csv(output_file, index=True)



def run_random_stdp(seed):
    output_file = f"accuracy_{seed}.txt"
    if os.path.exists(output_file):
        exit(0)
    np.random.seed(seed)
    accuracy = main(
        seed=np.random.randint(0, 0xFFFFFFF),
        duration_between_samples=100,
        input_leak_cst=np.exp(-1 / 30),
        output_leak_cst=np.exp(-1 / np.random.uniform(10, 100)),
        input_threshold=1,
        output_threshold=1,
        input_bias=0,
        refractory_period=np.random.randint(1, 5),
        lateral_inhibition_period=np.random.randint(1, 50),
        input_scale=np.random.uniform(0.001, 0.2),
        nb_epochs=1,
        use_vdsp=False,
        tau_pre=np.random.uniform(10, 100),
        tau_post=np.random.uniform(10, 100),
        dapre=np.abs(np.random.uniform(0, 1) * (10 ** np.random.randint(-5, -1))),
        dapost=np.abs(np.random.uniform(0, 1) * (10 ** np.random.randint(-5, -1))),
        norm_scale=np.random.uniform(10, 1000),
        weight_init_scale=np.random.uniform(0, 0.9),
        with_validation=False,
    )[0]
    with open(output_file, "w") as f:
        f.write(str(accuracy))


def run_vdsp(
    output_threshold=10,
    **args,
):
    return main(
        output_threshold=output_threshold,
        **args,
    )


def run_stdp(refractory_period=2, use_vdsp=False, lateral_inhibition_period=8, **args):
    return main(
        refractory_period=refractory_period,
        use_vdsp=use_vdsp,
        lateral_inhibition_period=lateral_inhibition_period,
        **args,
    )


def run_vdsp_vs_freq(seed, iteration, n_samples, normalize_duration=True):
    input_scale = np.logspace(-3.05, 0, n_samples)[iteration]
    acc, max_freq = run_vdsp(seed=seed, input_scale=input_scale, normalize_duration=normalize_duration)
    with open(f"vdsp_vs_freq_{iteration}_of_{n_samples}_seed_{seed}.txt", "w") as f:
        f.write(f"{acc}\n{max_freq}\n")


def run_stdp_vs_freq(seed, iteration, n_samples, normalize_duration=True):
    input_scale = np.logspace(-3.05, 0, n_samples)[iteration]
    acc, max_freq = run_stdp(seed=seed, input_scale=input_scale, normalize_duration=normalize_duration)
    with open(f"stdp_vs_freq_{iteration}_of_{n_samples}_seed_{seed}.txt", "w") as f:
        f.write(f"{acc}\n{max_freq}\n")


if __name__ == "__main__":
    Fire()
