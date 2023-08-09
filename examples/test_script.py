from torch.cuda import is_available as cuda_is_available
from torch.backends.mps import is_available as mps_is_available


def main():
    print(f"torch.cuda.is_available(): {cuda_is_available()}")
    print(f"torch.backends.mps.is_available(): {mps_is_available()}")


if __name__ == '__main__':
    main()
