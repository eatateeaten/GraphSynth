# Neural Network Graph Editor - Frontend

## Setup Instructions for Mac

1. First, install Node.js:
   - Open Terminal (press Cmd + Space, type "Terminal")
   - Install Homebrew if you don't have it:
     ```bash
     /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
     ```
   - Install Node.js:
     ```bash
     brew install node
     ```

2. Install pnpm:
   ```bash
   curl -fsSL https://get.pnpm.io/install.sh | sh -
   ```
   - Close and reopen Terminal for changes to take effect

3. Install project dependencies:
   - Open Terminal
   - Navigate to the project folder:
     ```bash
     cd path/to/nn-graph/front
     ```
   - Install dependencies:
     ```bash
     pnpm install
     ```
   - This might take a few minutes

4. Start the development server:
   ```bash
   pnpm dev
   ```
   - You should see a message saying the server is running
   - The site will automatically open in your browser
   - If it doesn't, go to: http://localhost:5173

## Common Issues

- If you see "command not found: pnpm":
  - Try closing and reopening Terminal
  - If that doesn't work, run the pnpm install command again

- If you see errors about missing dependencies:
  - Try running `pnpm install` again
  - Make sure you're in the correct folder (front)
