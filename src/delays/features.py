# 3. Feature engineering
def add_features(data, gare_freq):
    data = data.copy()
    data['holidays'] = data['date'].between('2023-06-01', '2023-08-31').astype(int)
    data['quarter'] = data['date'].dt.quarter
    data['gare_freq'] = data['gare'].map(gare_freq)
    data['propagation_index'] = (
        data['p2q0'] * 0.5 + data['p3q0'] * 0.3 + data['p4q0'] * 0.2
    )
    data['propagation_index_2'] = (
        data['p0q2'] * 0.5 + data['p0q3'] * 0.3 + data['p0q4'] * 0.2
        )
    data['gare_arret'] = data['gare'].astype(str) + '_' + data['arret'].astype(str)
    return data

