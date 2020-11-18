using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using Newtonsoft.Json;

namespace SimpleWebRequest
{
    class Program
    {
        static void Main(string[] args)
        {
            while (true)
            {
                WebRequest request = WebRequest.Create(
                "https://finnhub.io/api/v1/quote?symbol=AAPL&token=bu83akv48v6ufhqj8n10");

                WebResponse response = request.GetResponse();
            
                using (Stream dataStream = response.GetResponseStream())
                {
                    // Open the stream using a StreamReader for easy access.
                    StreamReader reader = new StreamReader(dataStream);
                    // Read the content.
                    string responseFromServer = reader.ReadToEnd();
                    // Display the content.

                    CurrentPrice result = JsonConvert.DeserializeObject<CurrentPrice>(responseFromServer);

                    Console.WriteLine(result.current_price);
                }
                System.Threading.Thread.Sleep(60000);
            }
          
        }

        public partial class CurrentPrice
        {
            [JsonProperty("c")]
            public double current_price { get; set; }

            [JsonProperty("h")]
            public double high_price { get; set; }

            [JsonProperty("l")]
            public double low_price { get; set; }

            [JsonProperty("o")]
            public double open_price { get; set; }

            [JsonProperty("pc")]
            public double close_price { get; set; }

            [JsonProperty("t")]
            public string time { get; set; }

        }
    }

    public class PPO
    {

        public PPO(
            //object state_dim,
            int action_dim,
            int n_latent_var,
            float learningRate,
            List<float> betas,
            float gamma,
            int K_epochs,
            float eps_clip)
        {
            var learning_rate = learningRate;
            this.betas = betas;
            this.gamma = gamma;
            this.eps_clip = eps_clip;
            this.K_epochs = K_epochs;
            this.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device);
            this.optimizer = torch.optim.Adam(this.policy.parameters(), lr: learningRate, betas: betas);
            this.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device);
            this.policy_old.load_state_dict(this.policy.state_dict());
            this.MseLoss = nn.MSELoss();
        }

    }
