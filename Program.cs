using credit_card_fraud;

internal class Program
{
    static void Main()
    {
        // OPTIONAL: Uncomment the following call to train the model again.
        // Run from the project folder containing credit_card_fraud_1000.csv.
        // This runs AutoML and replaces MLModel.zip with the newly trained model.
        // Comment the call again afterward to use the saved model without retraining.
        // MLModel.Train(
        //     outputModelPath: "MLModel.zip",
        //     inputDataFilePath: "credit_card_fraud_1000.csv");

        var input = new MLModel.ModelInput
        {
            TransactionId = 1,
            TransactionAmount = 168.73f,
            HourOfDay = 20,
            MerchantCategory = "Electronics",
            EntryMode = "Chip",
            Country = "CA",
            IsInternational = 1,
            CardPresent = 1,
            DaysSinceLastTransaction = 0.22f,
            TransactionCount24h = 6,
            AvgAmount7d = 33.34f
        };

        Console.WriteLine("Prediction Input Data");
        Console.WriteLine($"TransactionId: {input.TransactionId}");
        Console.WriteLine($"TransactionAmount: {input.TransactionAmount}");
        Console.WriteLine($"HourOfDay: {input.HourOfDay}");
        Console.WriteLine($"MerchantCategory: {input.MerchantCategory}");
        Console.WriteLine($"EntryMode: {input.EntryMode}");
        Console.WriteLine($"Country: {input.Country}");
        Console.WriteLine($"IsInternational: {input.IsInternational}");
        Console.WriteLine($"CardPresent: {input.CardPresent}");
        Console.WriteLine($"DaysSinceLastTransaction: {input.DaysSinceLastTransaction}");
        Console.WriteLine($"TransactionCount24h: {input.TransactionCount24h}");
        Console.WriteLine($"AvgAmount7d: {input.AvgAmount7d}");

        var result = MLModel.Predict(input);

        Console.WriteLine();
        Console.WriteLine("Prediction Result");
        Console.WriteLine($"Predicted label: {(result.PredictedLabel ? "Fraud" : "Not Fraud")}");
        Console.WriteLine($"Fraud probability: {result.Probability:P2}");
    }
}
