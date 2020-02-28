def A(x):
    def B(y):
        return y + z

    z = 4

    return B(x) + x


d = A(4)
