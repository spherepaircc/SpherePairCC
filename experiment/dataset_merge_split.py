import os
import sys
import hashlib

############################################
# 1. General file splitting and merging functions
############################################

def split_file(file_path, part_count=2):
    """
    General file splitting function that splits the file at file_path into part_count parts.
    :param file_path: The path of the file to be split
    :param part_count: Number of parts to split into, default is 2
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find the original file: {file_path}")

    file_size = os.path.getsize(file_path)
    part_size = file_size // part_count

    with open(file_path, 'rb') as f:
        for i in range(1, part_count + 1):
            part_file = f"{file_path}.part{i}"
            with open(part_file, 'wb') as pf:
                if i == part_count:
                    # Write the remaining bytes in the last part
                    pf.write(f.read())
                else:
                    pf.write(f.read(part_size))
            print(f"Created partial file: {part_file}")


def merge_file(file_path, part_count=2):
    """
    General file merging function that merges the part files back into the original file at file_path.
    :param file_path: The path of the merged file
    :param part_count: Number of part files to merge
    """
    with open(file_path, 'wb') as outfile:
        for i in range(1, part_count + 1):
            part_file = f"{file_path}.part{i}"
            if not os.path.exists(part_file):
                raise FileNotFoundError(f"Could not find partial file: {part_file}")
            with open(part_file, 'rb') as pf:
                outfile.write(pf.read())
            print(f"Merged partial file: {part_file}")
    print(f"Successfully merged to: {file_path}")


def calculate_md5(file_path):
    """
    Compute the MD5 checksum of the specified file.
    
    :param file_path: File path
    :return: MD5 checksum string, or None if the file does not exist
    """
    if not os.path.exists(file_path):
        return None

    md5 = hashlib.md5()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()


############################################
# 2. Splitting and merging CIFAR dataset
############################################

def split_cifar(dataset_name, part_count=2):
    """
    Split CIFAR dataset:
      ./dataset/{dataset_name}/{dataset_name}_pretrained_dataset.pkl
    Generates:
      {dataset_name}_pretrained_dataset.pkl.part1
      {dataset_name}_pretrained_dataset.pkl.part2
    """
    original_file = f"./dataset/{dataset_name}/{dataset_name}_pretrained_dataset.pkl"
    split_file(original_file, part_count=part_count)


def merge_cifar(dataset_name, part_count=2):
    """
    Merge CIFAR dataset:
      {dataset_name}_pretrained_dataset.pkl.part1
      {dataset_name}_pretrained_dataset.pkl.part2
    Produces:
      ./dataset/{dataset_name}/{dataset_name}_pretrained_dataset.pkl
    """
    original_file = f"./dataset/{dataset_name}/{dataset_name}_pretrained_dataset.pkl"
    merge_file(original_file, part_count=part_count)


def verify_cifar(dataset_name):
    """
    Verify the MD5 checksum of a CIFAR dataset
    """
    original_file = f"./dataset/{dataset_name}/{dataset_name}_pretrained_dataset.pkl"
    md5_val = calculate_md5(original_file)
    if md5_val is None:
        print(f"File does not exist: {original_file}")
    else:
        print(f"The MD5 checksum of {original_file} is: {md5_val}")


############################################
# 3. Splitting and merging Reuters dataset
############################################

def split_reuters(part_count=2):
    """
    The Reuters dataset includes two files:
      1. ./dataset/reuters/reutersidf10k_train.npy
      2. ./dataset/reuters/reutersidf10k_test.npy
    Split each file into:
      reutersidf10k_train.npy.part1, reutersidf10k_train.npy.part2
      reutersidf10k_test.npy.part1,  reutersidf10k_test.npy.part2
    """
    train_file = "./dataset/reuters/reutersidf10k_train.npy"
    test_file  = "./dataset/reuters/reutersidf10k_test.npy"

    print("Splitting Reuters training file...")
    split_file(train_file, part_count=part_count)
    print("Splitting Reuters test file...")
    split_file(test_file, part_count=part_count)


def merge_reuters(part_count=2):
    """
    Merge Reuters dataset split files:
      reutersidf10k_train.npy.part1, reutersidf10k_train.npy.part2 -> reutersidf10k_train.npy
      reutersidf10k_test.npy.part1,  reutersidf10k_test.npy.part2  -> reutersidf10k_test.npy
    """
    train_file = "./dataset/reuters/reutersidf10k_train.npy"
    test_file  = "./dataset/reuters/reutersidf10k_test.npy"

    print("Merging Reuters training file...")
    merge_file(train_file, part_count=part_count)
    print("Merging Reuters test file...")
    merge_file(test_file, part_count=part_count)


def verify_reuters():
    """
    Verify the MD5 checksums for the Reuters dataset:
      1. ./dataset/reuters/reutersidf10k_train.npy
      2. ./dataset/reuters/reutersidf10k_test.npy
    Compute and print MD5 for each file.
    """
    train_file = "./dataset/reuters/reutersidf10k_train.npy"
    test_file  = "./dataset/reuters/reutersidf10k_test.npy"
    
    train_md5 = calculate_md5(train_file)
    test_md5 = calculate_md5(test_file)
    
    if train_md5 is None:
        print(f"File does not exist: {train_file}")
    else:
        print(f"The MD5 checksum of {train_file} is: {train_md5}")
    
    if test_md5 is None:
        print(f"File does not exist: {test_file}")
    else:
        print(f"The MD5 checksum of {test_file} is: {test_md5}")


############################################
# 4. Main function entry
############################################

def main():
    """
    Based on user input, decide whether to perform split/merge/verify operations.
    This handles:
      1) CIFAR10: ./dataset/cifar10/cifar10_pretrained_dataset.pkl
      2) CIFAR100: ./dataset/cifar100/cifar100_pretrained_dataset.pkl
      3) Reuters: ./dataset/reuters/reutersidf10k_train.npy & reutersidf10k_test.npy

    Usage:
      python cifar_split_merge.py split
      python cifar_split_merge.py merge
      python cifar_split_merge.py verify
    """
    if len(sys.argv) != 2:
        print("Usage: python cifar_split_merge.py [split|merge|verify]")
        sys.exit(1)

    action = sys.argv[1].lower()
    part_count = 2  # Change this if you want more parts

    if action == 'split':
        print("==== Splitting CIFAR10 data ====")
        try:
            split_cifar('cifar10', part_count=part_count)
        except FileNotFoundError as e:
            print(e)
        print("")

        print("==== Splitting CIFAR100 data ====")
        try:
            split_cifar('cifar100', part_count=part_count)
        except FileNotFoundError as e:
            print(e)
        print("")

        print("==== Splitting Reuters data ====")
        try:
            split_reuters(part_count=part_count)
        except FileNotFoundError as e:
            print(e)

    elif action == 'merge':
        print("==== Merging CIFAR10 data ====")
        try:
            merge_cifar('cifar10', part_count=part_count)
        except FileNotFoundError as e:
            print(e)
        print("")

        print("==== Merging CIFAR100 data ====")
        try:
            merge_cifar('cifar100', part_count=part_count)
        except FileNotFoundError as e:
            print(e)
        print("")

        print("==== Merging Reuters data ====")
        try:
            merge_reuters(part_count=part_count)
        except FileNotFoundError as e:
            print(e)

    elif action == 'verify':
        print("==== Verifying CIFAR10 file integrity ====")
        verify_cifar('cifar10')
        print("")

        print("==== Verifying CIFAR100 file integrity ====")
        verify_cifar('cifar100')
        print("")

        print("==== Verifying Reuters file integrity ====")
        verify_reuters()

    else:
        print("Invalid action. Please use 'split', 'merge' or 'verify'.")
        print("Usage: python cifar_split_merge.py [split|merge|verify]")
        sys.exit(1)


if __name__ == "__main__":
    main()
