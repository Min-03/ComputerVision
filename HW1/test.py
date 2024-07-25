def test(arr):
    for x in arr:
        for j in range(10):
            print(x)
            x = x * 3
    return arr

print(test([1, 2, 3, 4]))
