
import numpy as np
import matplotlib.pyplot as plt
import math
from QuanTest2 import QuantizationTest2


def read_signal_from_txt():
    filename = input(
        "Enter input TXT filename (e.g. Quan2_input.txt): "
    ).strip().strip('"')

    try:
        with open(filename, "r", encoding="utf-8-sig") as file:
            lines = file.readlines()

        # Skip the first 3 metadata lines
        # Read the signal values from the second column
        samples = []

        for line in lines[3:]:
            parts = line.split()

            if len(parts) >= 2:
                samples.append(float(parts[1]))

        if len(samples) < 1:
            raise ValueError("No signal samples found in the file.")

        if not np.all(np.isfinite(samples)):
            raise ValueError("All samples must be finite numbers.")

        return np.array(samples, dtype=float)

    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return None

    except ValueError as error:
        print(f"Error reading signal: {error}")
        return None


def quantize_signal(samples, mode, value):
    signal = np.array(samples, dtype=float)

    # Calculate levels and bits
    if mode == "bits":
        bits = int(value)

        if bits < 1:
            raise ValueError("Number of bits must be at least 1.")

        levels = 2 ** bits

    else:
        levels = int(value)

        if levels < 2:
            raise ValueError("Number of levels must be at least 2.")

        bits = math.ceil(math.log2(levels))

    # Find minimum and maximum
    x_min = np.min(signal)
    x_max = np.max(signal)

    if x_min == x_max:
        delta = 0.0
        interval_indices = [1] * len(signal)
        quantized = signal.copy()
        sampled_error = quantized - signal
        encoded = ["0" * bits for _ in signal]

    else:
        # Quantization step size
        delta = (x_max - x_min) / levels

        # Calculate zero-based interval indices
        zero_based_indices = np.floor(
            (signal - x_min) / delta
        ).astype(int)

        zero_based_indices = np.clip(
            zero_based_indices, 0, levels - 1
        )

        # Test 2 expects interval indices starting from 1
        interval_indices = (
            zero_based_indices + 1
        ).tolist()

        # Calculate quantized values using interval midpoints
        quantized = (
            x_min + (zero_based_indices + 0.5) * delta
        )

        # Sampled quantization error
        sampled_error = quantized - signal

        # Encode the zero-based indices in binary
        encoded = [
            format(int(index), f"0{bits}b")
            for index in zero_based_indices
        ]

    return (
        signal,
        interval_indices,
        encoded,
        quantized,
        sampled_error,
        levels,
        bits,
        delta
    )


def display_results(
    signal,
    interval_indices,
    encoded,
    quantized,
    sampled_error,
    levels,
    bits,
    delta
):
    print("\n========== QUANTIZATION RESULTS ==========")
    print(f"Number of levels: {levels}")
    print(f"Number of bits: {bits}")
    print(f"Quantization step size: {delta:.6f}")

    print(
        "\nSample\tOriginal\tInterval Index\tEncoded"
        "\tQuantized\tSampled Error"
    )
    print("-" * 88)

    for i in range(len(signal)):
        print(
            f"{i}\t{signal[i]:.4f}\t\t"
            f"{interval_indices[i]}\t\t"
            f"{encoded[i]}\t"
            f"{quantized[i]:.4f}\t\t"
            f"{sampled_error[i]:.4f}"
        )

    # Plot original and quantized signals
    n = np.arange(len(signal))

    plt.figure(figsize=(11, 8))

    plt.subplot(2, 1, 1)

    plt.plot(
        n, signal, "bo-",
        label="Original Signal"
    )

    plt.step(
        n, quantized,
        where="mid",
        color="red",
        label="Quantized Signal"
    )

    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.title("Original vs Quantized Signal")
    plt.legend()
    plt.grid(True)

    # Plot sampled quantization error
    plt.subplot(2, 1, 2)

    plt.stem(
        n, sampled_error,
        linefmt="g-",
        markerfmt="go",
        basefmt="k-"
    )

    plt.axhline(0, color="black", linewidth=0.8)
    plt.xlabel("Sample Index")
    plt.ylabel("Sampled Error")
    plt.title("Quantization Error")
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    print("===== DIGITAL SIGNAL QUANTIZATION - TEST 2 =====")

    signal = read_signal_from_txt()

    if signal is None:
        return

    print("\nOriginal signal:", signal.tolist())

    print("\nChoose quantization method:")
    print("1. Number of levels")
    print("2. Number of bits")

    while True:
        choice = input("Your choice (1 or 2): ").strip()

        if choice in ("1", "2"):
            break

        print("Please enter 1 or 2.")

    while True:
        try:
            if choice == "1":
                value = int(
                    input("Enter number of levels: ")
                )

                if value < 2:
                    print("Number of levels must be at least 2.")
                    continue

                mode = "levels"

            else:
                value = int(
                    input("Enter number of bits: ")
                )

                if value < 1:
                    print("Number of bits must be at least 1.")
                    continue

                mode = "bits"

            break

        except ValueError:
            print("Please enter a valid integer.")

    (
        signal,
        interval_indices,
        encoded,
        quantized,
        sampled_error,
        levels,
        bits,
        delta
    ) = quantize_signal(signal, mode, value)

    display_results(
        signal,
        interval_indices,
        encoded,
        quantized,
        sampled_error,
        levels,
        bits,
        delta
    )

    # Run Test 2
    run_test = input(
        "\nRun QuantizationTest2? (y/n): "
    ).strip().lower()

    if run_test == "y":
        expected_file = input(
            "Enter expected-output filename "
            "(e.g. Quan2_Out.txt): "
        ).strip().strip('"')

        try:
            QuantizationTest2(
                expected_file,
                interval_indices,
                encoded,
                quantized.tolist(),
                sampled_error.tolist()
            )

        except FileNotFoundError:
            print(
                f"Test output file not found: {expected_file}"
            )


if __name__ == "__main__":
    main()
