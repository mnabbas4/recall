# APMA - AI Project Memory Assistant

🧠 An intelligent system for storing, querying, and learning from project problems and solutions using AI-powered semantic search.

## Features

- 📤 **Upload & Store**: CSV/Excel file upload or manual row-by-row entry
- 🔍 **Smart Search**: Semantic search using OpenAI embeddings
- 📊 **Insights**: Automated problem aggregation and pattern detection
- 🤖 **AI Narratives**: LLM-generated actionable insights
- 👥 **User Management**: Secure authentication system
- 🏷️ **Tracking**: All entries tagged with user ID

## Deployment on Streamlit Cloud

### Prerequisites

1. GitHub account
2. OpenAI API key ([Get one here](https://platform.openai.com/api-keys))
3. Streamlit Cloud account ([Sign up here](https://share.streamlit.io))

### Step-by-Step Deployment

#### 1. Prepare Your Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit - APMA deployment"

# Create repository on GitHub (https://github.com/new)
# Then connect and push:
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
git branch -M main
git push -u origin main
```

#### 2. Deploy to Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Select your GitHub repository
4. Set **Main file path**: `apma_app.py`
5. Click **"Deploy"**

#### 3. Configure Secrets

After deployment:

1. In Streamlit Cloud dashboard, click your app
2. Go to **Settings** → **Secrets**
3. Add the following:

```toml
OPENAI_API_KEY = "sk-proj-your-actual-key-here"
```

4. Click **"Save"**
5. App will automatically restart with the new configuration

### Important Notes

⚠️ **Data Persistence**: Streamlit Cloud resets data on app restart. For production:
- Consider connecting to external storage (AWS S3, Google Cloud Storage, PostgreSQL)
- Or use Streamlit's persistent storage solutions

⚠️ **Resource Limits**: Free tier has:
- 1GB RAM
- Shared CPU
- 200MB file upload limit

## Local Development

### Installation

```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-api-key-here"
```

Or set environment variable:

```bash
# Linux/Mac
export OPENAI_API_KEY="your-api-key-here"

# Windows
set OPENAI_API_KEY=your-api-key-here
```

### Run Locally

```bash
streamlit run apma_app.py
```

App will open at `http://localhost:8501`

## Project Structure

```
apma/
├── apma_app.py              # Main application
├── modules/
│   ├── __init__.py
│   ├── data_handler.py      # CSV/Excel validation
│   ├── embeddings_engine.py # OpenAI embeddings
│   ├── file_manager.py      # Memory file management
│   ├── recall_engine.py     # Semantic search engine
│   ├── user_manager.py      # Authentication
│   └── utils.py             # Utilities
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── requirements.txt         # Python dependencies
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

## Usage Guide

### 1. Create Account / Login

- Use sidebar to create account or login
- Each user gets a unique ID
- All entries are tagged with user ID

### 2. Upload Data

**Option A: File Upload**
- Upload CSV or Excel file
- Required columns: Project Category, Project Reference, Phase, Problems Encountered, Solutions Adopted
- Choose: Create new / Append to existing / Edit existing

**Option B: Manual Entry**
- Add rows one-by-one using the form
- Build up a buffer, then save to memory

### 3. Query Knowledge Base

- Select a memory to query
- Enter natural language query
- Adjust similarity threshold and other settings
- View results with scores and insights
- Generate AI narrative for actionable recommendations

### 4. Settings

- View saved memories
- Rebuild embeddings for all memories
- Check embeddings engine status

## Security Notes

🔒 **Never commit API keys to GitHub!**

- Keys are stored in Streamlit secrets or environment variables
- `.gitignore` prevents accidental commits
- Use `.streamlit/secrets.toml` for local development (already gitignored)

## Troubleshooting

### "Embeddings engine not active"
- Add OPENAI_API_KEY in Streamlit Cloud secrets
- Or set environment variable locally

### "No memories found"
- Upload a file first or add manual entries
- Check data/ directory exists

### "Embeddings not found"
- Click "Rebuild embeddings" in Settings
- This happens if embeddings were not computed during upload

### Data disappears after restart (Streamlit Cloud)
- This is expected on free tier
- Implement external storage for production

## Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - feel free to use for your projects!

## Support

For issues or questions:
- Open an issue on GitHub
- Check Streamlit documentation: https://docs.streamlit.io

---

Built with ❤️ using Streamlit and OpenAI
