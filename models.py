import torch
import torch.nn as nn
from torchvision import models

class SituationRecognizer(nn.Module):
    def __init__(self, num_verbs, num_roles, num_nouns, embedding_dim=300):
        """
        Initializes the model architecture with a memory-efficient role predictor.
        
        Args:
            num_verbs (int): The total number of unique verbs in the dataset.
            num_roles (int): The total number of unique argument roles across all verbs.
            num_nouns (int): The total number of unique nouns (potential arguments).
            embedding_dim (int): The size of the intermediate noun embeddings.
        """
        super().__init__()
        
        # 1. CNN Backbone
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone_out_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        for param in self.backbone.parameters():
            param.requires_grad = False
            
        # 2. Noun Embeddings
        # This layer stores a learnable vector for every noun in the dataset.
        self.noun_embeddings = nn.Embedding(num_embeddings=num_nouns, embedding_dim=embedding_dim)
        
        # 3. Prediction Heads
        self.verb_predictor = nn.Linear(in_features=backbone_out_features, out_features=num_verbs)
        
        # This NEW head predicts an "ideal" noun embedding for each role.
        # Its output size is now manageable: num_roles * embedding_dim
        self.role_embedding_predictor = nn.Linear(
            in_features=backbone_out_features, 
            out_features=num_roles * embedding_dim
        )
        
        self.num_roles = num_roles
        self.embedding_dim = embedding_dim

    def forward(self, image_batch):
        """
        Defines the forward pass with the new embedding-based logic.
        
        Args:
            image_batch (torch.Tensor): A batch of images of shape (N, 3, 224, 224).
            
        Returns:
            dict: A dictionary containing the verb logits and the role-noun logits.
        """
        # Extract features using the backbone
        # Output shape: (N, 2048)
        features = self.backbone(image_batch)
        
        # Predict verb scores (this remains the same)
        # Output shape: (N, num_verbs)
        verb_logits = self.verb_predictor(features)
        
        # Predict the ideal embedding for each role
        # Raw output shape: (N, num_roles * embedding_dim)
        predicted_role_embeddings_flat = self.role_embedding_predictor(features)
        
        # Reshape to (N, num_roles, embedding_dim)
        predicted_role_embeddings = predicted_role_embeddings_flat.view(-1, self.num_roles, self.embedding_dim)
        
        # Calculate logits by comparing predicted embeddings to all actual noun embeddings
        # We use a matrix multiplication (dot product) for this.
        # This is the key step for memory efficiency.
        #
        #   predicted_role_embeddings: (N, num_roles, embedding_dim)
        #   self.noun_embeddings.weight.t(): (embedding_dim, num_nouns)
        #
        # Resulting shape: (N, num_roles, num_nouns)
        role_logits = torch.matmul(predicted_role_embeddings, self.noun_embeddings.weight.t())
        
        return {
            'verb_logits': verb_logits,
            'role_logits': role_logits
        }

if __name__ == '__main__':
    # A simple test to verify the model's architecture and output shapes
    
    # Dummy parameters
    NUM_VERBS = 504
    NUM_ROLES = 211
    NUM_NOUNS = 17095
    BATCH_SIZE = 4

    # Instantiate the model
    model = SituationRecognizer(num_verbs=NUM_VERBS, num_roles=NUM_ROLES, num_nouns=NUM_NOUNS)
    print("Model instantiated successfully.")
    
    # Create a dummy batch of images
    dummy_images = torch.randn(BATCH_SIZE, 3, 224, 224)
    
    # Perform a forward pass
    with torch.no_grad():
        predictions = model(dummy_images)
        
    print("\n--- Output Shapes Verification ---")
    print(f"Input image batch shape: {dummy_images.shape}")
    
    verb_logits_shape = predictions['verb_logits'].shape
    role_logits_shape = predictions['role_logits'].shape
    
    print(f"Verb logits output shape: {verb_logits_shape}")
    print(f"Expected verb shape:     ({BATCH_SIZE}, {NUM_VERBS})")
    assert verb_logits_shape == (BATCH_SIZE, NUM_VERBS)
    
    print(f"Role logits output shape: {role_logits_shape}")
    print(f"Expected role shape:     ({BATCH_SIZE}, {NUM_ROLES}, {NUM_NOUNS})")
    assert role_logits_shape == (BATCH_SIZE, NUM_ROLES, NUM_NOUNS)
    
    print("\nModel test passed! The output shapes are correct.")
