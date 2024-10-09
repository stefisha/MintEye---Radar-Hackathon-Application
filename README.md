# Solana-Radar-Hackathon-DeFi-App 

The **MintEye** app was created for the Solana Radar Hackathon with the goal of creating a safer environment by providing a tool capable of detecting NFT scams. Hosted at [nft.stefisha.me](https://nft.stefisha.me/), the app utilizes two machine learning models: one focuses on analyzing images, while the other focuses on analyzing JSON metadata. Together, these models work to identify and flag risky NFTs, helping users make informed decisions.

## Project Overview

### Models:

#### 1. Image-Based Model
- **Purpose**: This model analyzes the visual aspects of NFTs to detect patterns and potential risks.
- **Outcome**: Generate insights based on visual features to flag suspicious NFTs or verify their legitimacy.

#### 2. Metadata-Based Model
- **Purpose**: This model focuses on analyzing various metadata fields to assess the legitimacy of NFTs.
- **Outcome**: Classify NFTs based on patterns or anomalies found in the metadata.

### Weighted Approach for Final Prediction:
- A **weighted average** is used to combine the results from the two models to generate a final prediction for each NFT.
  - **Outcome**: A unified classification indicating whether the NFT is legitimate or potentially fraudulent.

## Future Enhancements

- **NLP for Descriptions**: Implement natural language processing to analyze the descriptions for patterns associated with scams.
- **Caching for Performance**: Introduce caching mechanisms to improve the speed and performance of the prediction system.
- **Additional Improvements**: Expand the models to handle larger datasets and improve overall accuracy in future iterations.

This app aims to provide a robust solution for detecting scams in the growing NFT space, ensuring a safer environment for all users.
