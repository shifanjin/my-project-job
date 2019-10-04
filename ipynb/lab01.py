def summation(n):
    """Compute the summation i^3 + 3 * i^2 for 1 <= i <= n."""
    sum = 0
    for i in range(n) :
            sum = sum + pow(i, 3) + 3 * pow(i, 2)
    return sum

summation(1)


print("hi")
print(summation(1))
print(summation(2))
print(range(6))

for i in range(6):
    print(i)

for i in range(1,6):
    print(i)

print(2^5)
print(pow(2, 5))


ls = [1,2,3,4]
ls.append(5)
print(ls)
print(sum(ls))
