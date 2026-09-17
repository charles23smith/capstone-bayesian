import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

#using random data for now 
torch.manual_seed(42)
np.random.seed(42)

#3 steps to adapt this into diode workflow 
#1. replace synthetic data generator with real diode data 
#2. update input dimension of 1d-cnn in diode_encoder to match the length of the diode waveform
#3. featrure scaling and normalization of the diode waveform and parameters to improve training stability and convergence

def generate_random_data(num_shots = 10, time_steps = 100):
    t = np.linspace(0, 10, time_steps)
    waveforms, parameters, peak_voltages = [], [], []

    for _ in range(num_shots):
        #generate random parameters for waveform
        dose_rate = np.random.uniform(1.0, 10.0)
        load_impedance = np.random.uniform(10.0, 100.0)
        pulse_amplitude = np.random.uniform(0.5, 1.5) + 0.5 * dose_rate + (load_impedance / 50.0)
        voltage_curve = pulse_amplitude * np.exp(-0.4 * t) * np.sin(t) + np.random.normal(0,0.02, time_steps)
        v_peak = float(np.max(np.abs(voltage_curve)))

        waveforms.append(voltage_curve)
        parameters.append([dose_rate, load_impedance])
        peak_voltages.append(v_peak)

    return (torch.tensor(np.array(waveforms), dtype=torch.float32).unsqueeze(1), 
            torch.tensor(np.array(parameters), dtype=torch.float32), 
            torch.tensor(np.array(peak_voltages), dtype=torch.float32))


#defines encoder class for one arm of the simease network
class diode_encoder(nn.Module):
    def __init__(self, num_parameters=2, embedding_dim=16):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8)
        )

        self.fc = nn.Sequential(
            nn.Linear(16 * 8 + num_parameters, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )

    def forward(self, waveform, parameters):
        c_out = self.conv(waveform).view(waveform.size(0), -1)
        return self.fc(torch.cat((c_out, parameters), dim=1))

#siamese network class that takes two arms of the encoder and predicts the difference in peak voltage between two shots
class simease_network(nn.Module):
    def __init__(self, encoder, embedding_dim=16):
        super().__init__()
        self.encoder = encoder
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, wave_a, params_a, wave_b, params_b):
        h_diff = self.encoder(wave_a, params_a) - self.encoder(wave_b, params_b)
        return self.fc(h_diff).squeeze(-1)

#loops through N training shots to create N(N-1) pairs of training data for the simease network
def create_pairs(waveforms, parameters, peak_voltages):
    num_shots = len(waveforms)
    waveform_a, waveform_b, params_a, params_b, delta_v = [], [], [], [], []

    for i in range(num_shots):
        for j in range(num_shots):
            if i != j:
                waveform_a.append(waveforms[i])
                waveform_b.append(waveforms[j])
                params_a.append(parameters[i])
                params_b.append(parameters[j])
                delta_v.append(peak_voltages[i] - peak_voltages[j])

    return torch.stack(waveform_a), torch.stack(params_a), torch.stack(waveform_b), torch.stack(params_b), torch.tensor(delta_v)

def main():
    waves, params, peaks = generate_random_data(num_shots=10)
    train_w, test_w = waves[:8], waves[8:]
    train_p, test_p = params[:8], params[8:]
    train_y, test_y = peaks[:8], peaks[8:]

    w_a, p_a, w_b, p_b, delta_y = create_pairs(train_w, train_p, train_y)

    model = simease_network(diode_encoder())
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(1, 121):
        optimizer.zero_grad()
        loss = criterion(model(w_a, p_a, w_b, p_b), delta_y)
        loss.backward()
        optimizer.step()

    print(f"Final Pairwise Delta MSE Loss: {loss.item():.5f}\n")

    #create test inference for each test shot by comparing to all training shots and averaging the predicted peak voltage
    model.eval()
    with torch.no_grad():
        for i in range(len(test_y)):
            t_w, t_p = test_w[i].unsqueeze(0), test_p[i].unsqueeze(0)
            preds = []
            for r in range(len(train_y)):
                r_w, r_p = train_w[r].unsqueeze(0), train_p[r].unsqueeze(0)
                pred_delta = model(t_w, t_p, r_w, r_p).item()
                preds.append(train_y[r].item() + pred_delta)

            pred_y = np.mean(preds)
            actual = test_y[i].item()

            print(f"Test Shot #{i+1} [Dose Rate: {test_p[i][0]:.2f} Mrad/s, Load Z: {test_p[i][1]:.1f} ohms]")
            print(f"  Actual Peak Voltage (Max V) : {actual:.3f} V")
            print(f"  Predicted Peak Voltage      : {pred_y:.3f} V")
            print(f"  Error                       : {abs(actual - pred_y):.3f} V\n")

if __name__ == "__main__":
    main()



