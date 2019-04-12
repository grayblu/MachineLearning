import sim_data as sim_data
from sann import SNN2

x_data, y_data = sim_data.load_data()

model = SNN2(2,3)

model.fit(x_data, y_data, epochs=100, verbose=1)

loss, accuracy = model.evaluate(x_data, y_data, batch_size=100)

print(loss, accuracy)