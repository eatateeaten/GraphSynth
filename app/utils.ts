/**
 * Simple assertion function that throws an error if the condition is falsy
 */
export function assert(condition: any, message: string): asserts condition {
    if (!condition) {
        throw new Error(message);
    }
}

/**
 * Utility function to convert ANSI color codes in terminal output to HTML
 */
export function ansiToHtml(text: string): string {
    if (!text) return '';
    
    // Create a safe HTML version by escaping < and >
    let safeText = text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
    
    // Map of ANSI color codes to CSS classes
    const colorMap: Record<string, string> = {
        '0;30': 'color: #000000;', // black
        '0;31': 'color: #e74c3c;', // red
        '0;32': 'color: #2ecc71;', // green
        '0;33': 'color: #f39c12;', // yellow
        '0;34': 'color: #3498db;', // blue
        '0;35': 'color: #9b59b6;', // magenta
        '0;36': 'color: #1abc9c;', // cyan
        '0;37': 'color: #ecf0f1;', // white
        '1;30': 'color: #7f8c8d;', // bright black (gray)
        '1;31': 'color: #e74c3c; font-weight: bold;', // bright red
        '1;32': 'color: #2ecc71; font-weight: bold;', // bright green
        '1;33': 'color: #f39c12; font-weight: bold;', // bright yellow
        '1;34': 'color: #3498db; font-weight: bold;', // bright blue
        '1;35': 'color: #9b59b6; font-weight: bold;', // bright magenta
        '1;36': 'color: #1abc9c; font-weight: bold;', // bright cyan
        '1;37': 'color: #ffffff; font-weight: bold;', // bright white
    };

    // Stack to manage nested spans
    const stack: string[] = [];
    let result = '';
    
    // Parse text and convert ANSI codes to spans with inline styles
    const regex = /\u001b\[([0-9;]+)m/g;
    let lastIndex = 0;
    let match;
    
    while ((match = regex.exec(safeText)) !== null) {
        // Add text before the ANSI code
        result += safeText.substring(lastIndex, match.index);
        lastIndex = match.index + match[0].length;
        
        const code = match[1];
        
        // Handle reset code
        if (code === '0' || code === '0m') {
            // Close all open spans
            while (stack.length > 0) {
                result += '</span>';
                stack.pop();
            }
        } 
        // Handle color code
        else if (colorMap[code]) {
            result += `<span style="${colorMap[code]}">`;
            stack.push('span');
        }
    }
    
    // Add remaining text
    result += safeText.substring(lastIndex);
    
    // Close any remaining open spans
    while (stack.length > 0) {
        result += '</span>';
        stack.pop();
    }
    
    return result;
}
