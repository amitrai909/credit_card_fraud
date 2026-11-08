// AutoML API replaces Model Builder, as requested.
using Microsoft.ML;
using Microsoft.ML.AutoML;

namespace credit_card_fraud;

public partial class MLModel
{
    public const string RetrainFilePath = "credit_card_fraud_1000.csv";
    public const char RetrainSeparatorChar = ',';
    public const bool RetrainHasHeader = true;
    public const bool RetrainAllowQuoting = false;

    public static void Train(string outputModelPath, string inputDataFilePath = RetrainFilePath,
        char separatorChar = RetrainSeparatorChar, bool hasHeader = RetrainHasHeader,
        bool allowQuoting = RetrainAllowQuoting)
    {
        var mlContext = new MLContext();
        var data = LoadIDataViewFromFile(mlContext, inputDataFilePath, separatorChar, hasHeader, allowQuoting);
        var model = RetrainModel(mlContext, data);
        SaveModel(mlContext, model, data, outputModelPath);
    }

    public static IDataView LoadIDataViewFromFile(MLContext mlContext, string inputDataFilePath,
        char separatorChar, bool hasHeader, bool allowQuoting)
    {
        return mlContext.Data.LoadFromTextFile<ModelInput>(inputDataFilePath, separatorChar,
            hasHeader, allowQuoting: allowQuoting);
    }

    public static void SaveModel(MLContext mlContext, ITransformer model, IDataView data, string modelSavePath)
    {
        DataViewSchema dataViewSchema = data.Schema;
        using (var fs = File.Create(modelSavePath))
        {
            mlContext.Model.Save(model, dataViewSchema, fs);
        }
    }

    public static ITransformer RetrainModel(MLContext mlContext, IDataView trainData)
    {
        var columns = new ColumnInformation { LabelColumnName = nameof(ModelInput.Fraud) };
        columns.IgnoredColumnNames.Add(nameof(ModelInput.TransactionId));
        columns.CategoricalColumnNames.Add(nameof(ModelInput.MerchantCategory));
        columns.CategoricalColumnNames.Add(nameof(ModelInput.EntryMode));
        columns.CategoricalColumnNames.Add(nameof(ModelInput.Country));
        var experiment = mlContext.Auto().CreateBinaryClassificationExperiment(120);
        var result = experiment.Execute(trainData, columnInformation: columns);
        return result.BestRun.Model;
    }
}
