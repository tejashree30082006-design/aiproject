from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(resume_vectors, job_vector):
    similarity = cosine_similarity(resume_vectors, job_vector)
    return similarity.flatten()