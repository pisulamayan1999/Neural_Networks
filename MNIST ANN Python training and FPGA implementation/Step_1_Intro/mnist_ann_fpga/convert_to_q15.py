import numpy as np

def float_to_q15(x):
    # Q1.15 range: -1.0 to almost +1.0
    x = np.clip(x, -1.0, 0.999969482421875)
    return np.round(x * 32768).astype(np.int16)

files = [
    "hidden_weight.txt",
    "hidden_bias.txt",
    "output_weight.txt",
    "output_bias.txt"
]

for file in files:
    data = np.loadtxt(file)
    q15_data = float_to_q15(data)

    output_file = file.replace(".txt", "_q15.txt")
    np.savetxt(output_file, q15_data, fmt="%d")

    print(f"Converted {file} -> {output_file}")

print("All done.")