def find_common(a, b):
    result = []

    for x in a:
        for y in b:
            if x == y:
                if x not in result:
                    result.append(x)

    return result


def calculate_total(numbers):
    total = 0

    for i in range(len(numbers)):
        total = total + numbers[i]

    return total


def search_names(names, target):
    found = False

    for i in range(len(names)):
        if names[i] == target:
            found = True

    return found


def build_message(name, age):
    message = ""

    for i in range(1000):
        message = message + "Hello " + name + ", you are " + str(age)

    return message