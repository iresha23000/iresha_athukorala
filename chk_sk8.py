import cohere
import argparse

# Set up your Cohere API key
API_KEY = 'QV7Gzsyp3RjngMzlsZrGqdoxTHaRp9qMYdOjghCD'

# Initialize the Cohere client
co = cohere.Client(API_KEY)

def generate_creative_content(prompt, num_variations=3):
    """Generates creative writing based on the user prompt using Cohere API"""
    
    # Adjust parameters for creativity and variety
    temperature = 0.85  # Controls randomness: higher value creates more creative outputs
    top_p = 0.9         # Controls diversity via nucleus sampling
    presence_penalty = 0.6  # Encourages introducing new topics in the generations
    # Remove frequency_penalty to avoid conflict with presence_penalty

    # Generate content with 3 variations
    responses = co.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=300,  # Limit on the number of tokens in each response
        temperature=temperature,
        p=top_p,
        presence_penalty=presence_penalty,
        num_generations=num_variations  # Generate 3 different variations
    )
    
    # Return the generated variations
    return [response.text.strip() for response in responses.generations]


def main():
    # Set up the argument parser for command-line interaction
    parser = argparse.ArgumentParser(description='Creative Writer using Cohere API')
    parser.add_argument('prompt', type=str, help='The prompt to generate content from')
    parser.add_argument('--variations', type=int, default=3, help='Number of variations to produce (default: 3)')

    # Parse command-line arguments
    args = parser.parse_args()

    # Generate creative content based on the input prompt
    print("\nGenerating creative content...\n")
    content_variations = generate_creative_content(prompt=args.prompt, num_variations=args.variations)

    # Print the different versions
    for i, content in enumerate(content_variations, 1):
        print(f"--- Version {i} ---\n{content}\n")


if __name__ == "__main__":
    main()
