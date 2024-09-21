namespace VectorSearch
{
    using System;
    using System.Threading.Tasks;
    using System.Net.Http;
    using System.Text;
    using Azure.AI.OpenAI;
    using Azure;
    using System.Security.Claims;

    namespace AzureAISearchExample
    {
        public class Query
        {
            private static string openAiEndpoint = "https://dirthackopenai2.openai.azure.com/";
            private static string openAiApiKey = "";
            private static string modelDeploymentName = "dirthack2024openai2";

            static void Main(string[] args)
            {
                string queryText = "huge mace excavated south of india.";
                var openAiClient = new AzureOpenAIClient(new Uri(openAiEndpoint), new AzureKeyCredential(openAiApiKey));
                var embeddingclient = openAiClient.GetEmbeddingClient(modelDeploymentName);
                float[] vector = embeddingclient.GenerateEmbedding(queryText).Value.Vector.ToArray();
                string searchServiceName = "dirthack2024-std2";
                string indexName = "dirthackindex2";
                string apiKey = "";
                string uri = $"https://{searchServiceName}.search.windows.net/indexes/{indexName}/docs/search?api-version=2024-07-01";

                var query = new
                {
                    count = true,
                    select = "claim, url, explanation",
                    vectorQueries = new[]
                    {
                    new
                    {
                        kind = "vector",
                        vector = vector,
                        exhaustive = true,
                        fields = "vector",
                        weight = 0.5,
                        k = 5
                    }
                }
                };

                using (HttpClient client = new HttpClient())
                {
                    client.DefaultRequestHeaders.Add("api-key", apiKey);
                    var content = new StringContent(System.Text.Json.JsonSerializer.Serialize(query), Encoding.UTF8, "application/json");

                    HttpResponseMessage response = client.PostAsync(uri, content).Result;
                    string result = response.Content.ReadAsStringAsync().Result;

                    Console.WriteLine(result);
                }
            }
        }
    }
}
