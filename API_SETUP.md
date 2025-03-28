# Setting Up DeepSeek API Key

This guide explains how to securely set up your DeepSeek API key for the TI_RAG system.

## Setup Instructions

1. **Locate the template file**
   
   Find the `.env.template` file in the root directory of the project.

2. **Create your own .env file**
   
   ```bash
   cp .env .env
   ```

3. **Edit your .env file**
   
   Open the newly created `.env` file and add your DeepSeek API key:
   
   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

4. **Keep your .env file secure**
   
   - Never commit your `.env` file to version control
   - The `.gitignore` file is already set up to ignore `.env` files
   - Keep backups of your `.env` file in a secure location

## Obtaining a DeepSeek API Key

1. Register or login to your DeepSeek account
2. Navigate to the API key section in your account settings
3. Generate a new API key
4. Copy the key to your `.env` file

## Verifying Configuration

You can verify that your environment variables are properly loaded by running:

```bash
python3 -c "import os; from dotenv import load_dotenv; load_dotenv(); print(f'DeepSeek API Key configured: {bool(os.getenv(\"DEEPSEEK_API_KEY\"))}')"
```

This should output `DeepSeek API Key configured: True` if your key is properly set up.

## Troubleshooting

If you encounter issues with API access:

1. Verify that your `.env` file exists in the project root
2. Check that the variable name is exactly `DEEPSEEK_API_KEY`
3. Ensure your API key is valid and not expired
4. Restart your application after making changes to the `.env` file
