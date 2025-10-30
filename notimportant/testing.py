def user_function():
    base_amount = input("How much money you got? ")
    interest_rate = input("What's the interest rate (as a decimal)? ")
    saving_years = input("How many years do you plan to save? ")
    saving_rate = input("How much of your money do you plan to save each year (as a decimal)? ")
    total = calculate_amount(base_amount, interest_rate, saving_years, saving_rate)
    return total


def calculate_amount(base_amount, interest_rate, saving_years, yearly_saving):
    base_amount = float(base_amount)
    interest_rate = float(interest_rate)
    saving_years = int(saving_years)
    yearly_saving = float(yearly_saving)   # treat it as dollars, not %

    total_amount = base_amount
    for year in range(saving_years):
        total_amount = (total_amount + yearly_saving) * (1 + interest_rate)

    return total_amount
 

def count_diff(word):
    if not isinstance(word, str):
        raise TypeError(f"Expected a string, got {type(word).__name__}")
    else :
        i = 0
        count = 0
        word = word.lower()
        print(word)
        for i in range(len(word)-1):
            if word[i] != word[i+1]:
                count += 1
        print(count)
        return count
    
def loop():
    again = 'y'
    while again.lower() == 'y':
        word = input("Enter a word: ")
        count_diff(word)
        again = input("Do you want to try again? (y/n): ")

def maximum_triplet(arr):
    sorted_arr = sorted(arr, reverse=True)
    sum = 1
    print(sorted_arr)
    for i in range(3):
        sum = sum * sorted_arr[i]
    print(sum)
    return sum


def move(arr):
    for i in arr:
        if i == 0:
            arr.remove(i)
            arr.append(i)
    print(arr)
    
def cry():
    print("I am crying")
    return "I am crying"

def yeeees(arr):
    for i in arr:
        for j in arr:
            if i == j:
                arr.remove(j)
                print(i)

    print(arr)

def wave(arr):
    arr.sort()
    for i in range(0, len(arr)-1, 2):
        arr[i], arr[i+1] = arr[i+1], arr[i]
    print(arr)
    return arr

def reverse(String):
    str = ""
    for i in String:
        str = String[::-1]
    print(str)
    return str
        
def fifty(arr, correct_answer):
    import random
    random.shuffle(arr)
    arr.remove(correct_answer)
    print(f"Shuffled array: {arr}")
    arr.pop()
    arr.pop()
    arr = arr + [correct_answer]
    arr.sort()
    print(arr)



    fifty(["A","B","C", "D"], "D")

