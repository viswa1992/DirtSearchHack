using Azure;
using Azure.AI.OpenAI;
using Microsoft.AspNetCore.Mvc;
using Newtonsoft.Json;
using System.Text;

namespace BackendAPI.Controllers
{
    [ApiController]
    [Route("searchdirt")]
    public class DirtSearchController : ControllerBase
    {
        private static string openAiEndpoint = "https://dirthackopenai2.openai.azure.com/";
        private static string openAiApiKey = "";
        private static string modelDeploymentName = "dirthack2024openai2";

        private readonly ILogger<DirtSearchController> _logger;

        public DirtSearchController(ILogger<DirtSearchController> logger)
        {
            _logger = logger;
        }

        [HttpGet("{searchquery}")]
        public IEnumerable<DirtResponse> Get(string? searchquery)
        {
            var openAiClient = new AzureOpenAIClient(new Uri(openAiEndpoint), new AzureKeyCredential(openAiApiKey));
            var embeddingclient = openAiClient.GetEmbeddingClient(modelDeploymentName);
            float[] vector = embeddingclient.GenerateEmbedding(searchquery).Value.Vector.ToArray();
            string searchServiceName = "dirthack2024-std2";
            string indexName = "dirthackindex3";
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
                var resultStr = response.Content.ReadAsStringAsync().Result;
                var result = JsonConvert.DeserializeObject<Rootobject>(resultStr);
                foreach (var item in result.value)
                {
                    item.searchscore = item.searchscore - 1;
                    if (item.searchscore > 0.69)
                    {
                        item.matchConfidence = "High";
                    }
                    else if (item.searchscore > 0.59)
                    {
                        item.matchConfidence = "Medium";
                    }
                    else
                    {
                        item.matchConfidence = "Low";
                    }
                }
                if (result?.value != null && result.value.Length > 0)
                {
                    return result.value;
                }
                else
                {
                    return null;
                }
            }
        }
    }
}
