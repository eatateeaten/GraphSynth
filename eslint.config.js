import js from '@eslint/js';
import tsParser from '@typescript-eslint/parser';
import tsPlugin from '@typescript-eslint/eslint-plugin';
import reactPlugin from 'eslint-plugin-react';
import reactHooksPlugin from 'eslint-plugin-react-hooks';
import importPlugin from 'eslint-plugin-import';
import globals from 'globals';

export default [
    {
        ignores: ['**/node_modules/**', '**/env/**', '**/dist/**', '**/build/**', '**/old/**']
    },
    js.configs.recommended,
    {
        files: ['**/*.ts', '**/*.tsx'],
        languageOptions: {
            parser: tsParser,
            parserOptions: {
                ecmaVersion: 2021,
                sourceType: 'module',
                ecmaFeatures: {
                    jsx: true
                },
                project: './tsconfig.json',
                tsconfigRootDir: '.'
            },
            globals: {
                ...globals.browser,
                ...globals.node,
            }
        },
        linterOptions: {
            reportUnusedDisableDirectives: true
        },
        plugins: {
            '@typescript-eslint': tsPlugin,
            'react': reactPlugin,
            'react-hooks': reactHooksPlugin,
            'import': importPlugin
        },
        rules: {
            // Disable style/formatting rules
            'indent': 'off',
            'linebreak-style': 'off',
            'quotes': 'off',
            'semi': 'off',

            // Focus on important code quality issues
            'no-unused-vars': 'off',
            'no-case-declarations': 'off',
            'no-undef': 'error',
            'no-duplicate-imports': 'error',
            'no-constant-condition': 'error',
            'no-unreachable': 'error',
            'no-unsafe-negation': 'error',
            'no-unsafe-optional-chaining': 'error',
            'no-throw-literal': 'error',

            // Import/export rules
            'import/no-unresolved': 'error',
            'import/named': 'error',
            'import/default': 'error',
            'import/namespace': 'error',
            'import/no-restricted-paths': 'error',
            'import/no-cycle': 'error',

            // TypeScript specific important rules
            '@typescript-eslint/no-explicit-any': 'off',
            '@typescript-eslint/no-unused-vars': 'off',
            '@typescript-eslint/no-non-null-assertion': 'off',
            '@typescript-eslint/no-misused-promises': 'error',
            '@typescript-eslint/no-floating-promises': 'error',
            '@typescript-eslint/no-unnecessary-type-assertion': 'error',

            // React specific important rules
            'react/react-in-jsx-scope': 'off',
            'react/prop-types': 'off',
            'react-hooks/rules-of-hooks': 'error',
            'react-hooks/exhaustive-deps': 'warn'
        },
        settings: {
            'import/parsers': {
                '@typescript-eslint/parser': ['.ts', '.tsx']
            },
            'import/resolver': {
                typescript: {
                    alwaysTryTypes: true,
                    project: './tsconfig.json'
                }
            },
            react: {
                version: 'detect'
            }
        }
    },
    // Add Jest environment for test files
    {
        files: ['**/*.test.ts', '**/*.test.tsx', '**/test/**/*.ts', '**/test/**/*.tsx'],
        languageOptions: {
            globals: {
                ...globals.jest,
            }
        }
    },
    // Add specific configuration for Config JS files
    {
        files: ['jest.config.js', '*.config.js', 'vite.config.ts'],
        languageOptions: {
            globals: {
                module: 'writable',
                require: 'readonly',
                __dirname: 'readonly',
                process: 'readonly'
            }
        }
    }
];
