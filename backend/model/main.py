class Model:
    def __init__(self):
        pass

    def get_granulometry(self):
        import random
        value = round(random.uniform(50.0, 85.0), 2)  # Return random value between 50 and 85 with 2 decimal places
        print(f"MODEL: Generated granulometry value: {value}")
        return value