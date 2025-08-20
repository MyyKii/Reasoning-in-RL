from data_collector import collect_data

X, Y = collect_data(episodes=20, steps=300)
print("Trainingsdaten:", X.shape, Y.shape)
