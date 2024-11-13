from locust import HttpUser, task, between

class NFTUser(HttpUser):
    # Wait time between each task execution (in seconds)
    wait_time = between(1, 5)

    @task
    def fetch_nft(self):
        # Example mint address to test
        mint_address = "AxZoL86C9abaQFQ7BVzEvkw1BYthsZCdnR6LfZVtrRVJ"
        self.client.post("/fetch_nft", data={"nft_mint": mint_address})
