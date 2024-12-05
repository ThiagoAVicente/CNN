import csv

# Função para ler os dados do CSV e ignorar o cabeçalho
def read_csv(filename):
    data = []
    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Pula a primeira linha (cabeçalho)
        for row in reader:
            data.append(row)
    return data

# Função para calcular o melhor threshold para uma coluna específica
def find_best_threshold(data, column_index):
    # Converter valores da coluna para float
    column_values = [float(row[column_index]) for row in data if row[column_index].replace('.', '', 1).isdigit()]

    # Se não houver valores válidos, retornar 0 como threshold
    if not column_values:
        return 0

    min_value = min(column_values)
    max_value = max(column_values)

    best_threshold = 0
    best_accuracy = 0

    # Testar diferentes thresholds
    step = 0.01
    for threshold in frange(min_value, max_value, step):
        correct_predictions = 0
        total_predictions = 0

        for row in data:
            if row[column_index].replace('.', '', 1).isdigit():
                value = float(row[column_index])
                predicted_class = 1 if value > threshold else 0
                actual_class = int(row[-1])  # Supondo que a última coluna seja a classe

                if predicted_class == actual_class:
                    correct_predictions += 1
                total_predictions += 1

        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold

def frange(start, stop, step):
    while start < stop:
        yield round(start, 2)
        start += step

def main():
    filename = 'train.csv'  
    data = read_csv(filename)

    if not data:
        print("Erro ao ler os dados do arquivo CSV.")
        return

    num_fields = len(data[0]) - 1 
    for i in range(num_fields):
        best_threshold = find_best_threshold(data, i)
        print(f"Melhor threshold para o campo {i+1}: {best_threshold}")

if __name__ == '__main__':
    main()
