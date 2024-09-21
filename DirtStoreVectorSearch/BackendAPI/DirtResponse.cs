using Newtonsoft.Json;

namespace BackendAPI
{
    public class Rootobject
    {
        [JsonProperty("@odata.context")]
        public string odatacontext { get; set; }

        [JsonProperty("@odata.count")]
        public int odatacount { get; set; }

        public DirtResponse[] value { get; set; }
    }

    public class DirtResponse
    {
        [JsonProperty("@search.score")]
        public double searchscore { get; set; }
        public string matchConfidence { get; set; }

        public string claim { get; set; }
        public string explanation { get; set; }
        public string url { get; set; }
    }
}


