using Microsoft.Extensions.DependencyInjection;
using Radiate.Examples.Examples;

namespace Radiate.Examples;

public delegate IExample ExampleResolver(int exampleNumber);

public static class ServiceBuilder
{
    public static IServiceCollection AddServices(this IServiceCollection services) =>
        services
            .AddScoped<BlobMeans>()
            .AddScoped<BostonRegression>()
            .AddScoped<ConvNetMinst>()
            .AddScoped<EvolveForest>()
            .AddScoped<EvolveHelloWorld>()
            .AddScoped<EvolveNEAT>()
            .AddScoped<EvolveTree>()
            .AddScoped<RandomForestPredictor>()
            .AddScoped<SVMPredictor>()
            .AddScoped<TempuratureTimeSeries>()
            .AddScoped<TrainDense>()
            .AddScoped<TrainLSTM>()
            .AddScoped<ExampleResolver>(sp => num => num switch
            {
                1 => sp.GetRequiredService<EvolveNEAT>(),
                2 => sp.GetRequiredService<EvolveHelloWorld>(),
                3 => sp.GetRequiredService<TrainDense>(),
                4 => sp.GetRequiredService<TrainLSTM>(),
                5 => sp.GetRequiredService<BostonRegression>(),
                6 => sp.GetRequiredService<ConvNetMinst>(),
                7 => sp.GetRequiredService<BlobMeans>(),
                8 => sp.GetRequiredService<RandomForestPredictor>(),
                9 => sp.GetRequiredService<SVMPredictor>(),
                10 => sp.GetRequiredService<EvolveTree>(),
                11 => sp.GetRequiredService<EvolveForest>(),
                12 => sp.GetRequiredService<TempuratureTimeSeries>()
            });
}