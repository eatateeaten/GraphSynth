# Neural Network Graph Editor - Frontend

## Setup Instructions for Mac

1. First, install Node.js:
   - Open Terminal
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
   - Go to: http://localhost:5173
