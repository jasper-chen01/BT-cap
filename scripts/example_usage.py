"""
Example usage of the BAT Portal API
"""
import requests
import json

API_URL = "http://localhost:8000/api"


def check_health():
    """Check API health"""
    response = requests.get(f"{API_URL}/health")
    print("Health Check:")
    print(json.dumps(response.json(), indent=2))
    print()


def annotate_file(file_path: str, top_k: int = 10, similarity_threshold: float = 0.7):
    """Annotate a single-cell data file"""
    print(f"Annotating {file_path}...")
    
    with open(file_path, 'rb') as f:
        files = {'file': f}
        data = {
            'top_k': top_k,
            'similarity_threshold': similarity_threshold
        }
        response = requests.post(f"{API_URL}/annotate", files=files, data=data)
    
    if response.status_code == 200:
        result = response.json()
        print(f"Successfully annotated {result['total_cells']} cells")
        
        # Print summary statistics
        annotations = result['annotations']
        annotation_counts = {}
        for cell in annotations:
            ann = cell['predicted_annotation']
            annotation_counts[ann] = annotation_counts.get(ann, 0) + 1
        
        print("\nAnnotation Summary:")
        for ann, count in sorted(annotation_counts.items(), key=lambda x: -x[1]):
            print(f"  {ann}: {count} cells")
        
        return result
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None


if __name__ == "__main__":
    # Check health
    check_health()
    
    # Example: annotate a file (uncomment and provide path)
    # result = annotate_file("path/to/your/data.h5ad", top_k=10, similarity_threshold=0.7)
