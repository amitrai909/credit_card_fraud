// Authored using the supplied ZIP's partial-class consumption structure.
// This is not a Model Builder export; training uses the AutoML API as requested.
using Microsoft.ML;
using Microsoft.ML.Data;

namespace credit_card_fraud;

public partial class MLModel
{
    public class ModelInput
    {
        [LoadColumn(0)] public float TransactionId { get; set; }
        [LoadColumn(1)] public float TransactionAmount { get; set; }
        [LoadColumn(2)] public float HourOfDay { get; set; }
        [LoadColumn(3)] public string MerchantCategory { get; set; } = "";
        [LoadColumn(4)] public string EntryMode { get; set; } = "";
        [LoadColumn(5)] public string Country { get; set; } = "";
        [LoadColumn(6)] public float IsInternational { get; set; }
        [LoadColumn(7)] public float CardPresent { get; set; }
        [LoadColumn(8)] public float DaysSinceLastTransaction { get; set; }
        [LoadColumn(9)] public float TransactionCount24h { get; set; }
        [LoadColumn(10)] public float AvgAmount7d { get; set; }
        [LoadColumn(11)] public bool Fraud { get; set; }
    }

    public class ModelOutput
    {
        [ColumnName("PredictedLabel")] public bool PredictedLabel { get; set; }
        public float Probability { get; set; }
        public float Score { get; set; }
    }

    private static string MLNetModelPath = Path.GetFullPath("MLModel.zip");
    public static readonly Lazy<PredictionEngine<ModelInput, ModelOutput>> PredictEngine = new Lazy<PredictionEngine<ModelInput, ModelOutput>>(() => CreatePredictEngine(), true);

    private static PredictionEngine<ModelInput, ModelOutput> CreatePredictEngine()
    {
        var mlContext = new MLContext();
        var model = mlContext.Model.Load(MLNetModelPath, out _);
        return mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);
    }

    public static ModelOutput Predict(ModelInput input)
    {
        var predEngine = PredictEngine.Value;
        return predEngine.Predict(input);
    }
}
