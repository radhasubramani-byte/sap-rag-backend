import os

print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_SERVICE_KEY exists:", bool(os.getenv("SUPABASE_SERVICE_KEY")))
print("OPENAI_API_KEY exists:", bool(os.getenv("OPENAI_API_KEY")))
