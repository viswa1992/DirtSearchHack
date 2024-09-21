using System;
using System.IO;
using System.Net;
using System.Security.Claims;
using System.Threading.Tasks;
using Azure;
using Azure.AI.OpenAI;
using Azure.Search.Documents;
using Azure.Search.Documents.Indexes;
using Azure.Search.Documents.Indexes.Models;
using Azure.Search.Documents.Models;
using OpenAI.Embeddings;

class Ingest
{
    private static string openAiEndpoint = "https://dirthackopenai2.openai.azure.com/";
    private static string openAiApiKey = "";
    private static string searchServiceEndpoint = "https://dirthack2024-std2.search.windows.net";
    private static string searchServiceApiKey = "";
    private static string indexName = "dirthackindex2";
    private static string modelDeploymentName = "dirthack2024openai2";
    private static List<Document> documents = new List<Document>();

    static void Main(string[] args)
    {
        var openAiClient = new AzureOpenAIClient(new Uri(openAiEndpoint), new AzureKeyCredential(openAiApiKey));
        var searchClient = new SearchClient(new Uri(searchServiceEndpoint), indexName, new AzureKeyCredential(searchServiceApiKey));
        List<string> lines = File.ReadAllLines("C:\\Users\\viswajm\\Desktop\\Tasks\\hack2024\\ThreatIntel.tsv").ToList();
        lines.RemoveAt(0);
        int currentBatch = 0;
        foreach (var line in lines)
        {
            var columns = line.Split('\t');

            Document doc = new Document()
            {
                claim = columns[1],
                url = columns[3],
                explanation = columns[2],
                id = columns[0]
            };

            documents.Add(doc);
            ++currentBatch;

            if (currentBatch > 1000)
            {
                FlushData(openAiClient, searchClient);
                currentBatch = 0;
            }
        }
        FlushData(openAiClient, searchClient);
    }

    private static void FlushData(AzureOpenAIClient openAiClient, SearchClient searchClient)
    {
        var embeddings = GetEmbeddings(openAiClient, documents.Select(x => x.claim));
        for (int j = 0; j < documents.Count; j++)
        {
            documents[j].vector = embeddings.ElementAt(j);
        }

        var x = searchClient.UploadDocumentsAsync(documents.ToArray()).Result;
        documents.Clear();
    }

    private static IEnumerable<float[]> GetEmbeddings(AzureOpenAIClient client, IEnumerable<string> claims)
    {
        var embeddingclient = client.GetEmbeddingClient(modelDeploymentName);
        var result = embeddingclient.GenerateEmbeddings(claims).Value.Select(x => x.Vector.ToArray());
        return result;
    }


    class Document
    {
        [SimpleField(IsKey = true)]
        public string id { get; set; }
        public string claim { get; set; }
        public string url { get; set; }
        public string explanation { get; set; }
        public float[] vector { get; set; }
    }
}
