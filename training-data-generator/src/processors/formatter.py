"""Content formatting utilities."""

from typing import Dict, Any, List


class ConversationFormatter:
    """Format content into conversation structure."""

    @staticmethod
    def create_simple_conversation(question: str, answer: str) -> Dict[str, Any]:
        """Create a simple Q&A conversation.

        Args:
            question: User question
            answer: Assistant answer

        Returns:
            Conversation dictionary
        """
        return {
            'messages': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': answer}
            ]
        }

    @staticmethod
    def create_code_conversation(
        question: str,
        explanation: str,
        code: str,
        language: str = 'python',
        output: str = None
    ) -> Dict[str, Any]:
        """Create a conversation with code.

        Args:
            question: User question
            explanation: Explanation text
            code: Code content
            language: Programming language
            output: Optional code output

        Returns:
            Conversation dictionary
        """
        assistant_content = [
            {'type': 'text', 'text': explanation}
        ]

        if code:
            assistant_content.append({
                'type': language,
                'text': code
            })

        if output:
            assistant_content.append({
                'type': 'python_output',
                'text': output
            })

        return {
            'messages': [
                {'role': 'user', 'content': question},
                {'role': 'assistant', 'content': assistant_content}
            ]
        }

    @staticmethod
    def create_multi_turn_conversation(turns: List[tuple]) -> Dict[str, Any]:
        """Create a multi-turn conversation.

        Args:
            turns: List of (role, content) tuples

        Returns:
            Conversation dictionary
        """
        messages = []
        for role, content in turns:
            messages.append({
                'role': role,
                'content': content
            })

        return {'messages': messages}


class PRFormatter:
    """Format GitHub PR data into conversations."""

    @staticmethod
    def format_pr_as_conversation(
        pr_title: str,
        pr_description: str,
        commits: List[Dict[str, Any]],
        parse_commits: bool = True
    ) -> Dict[str, Any]:
        """Format PR data as a conversation.

        Args:
            pr_title: PR title
            pr_description: PR description
            commits: List of commit data
            parse_commits: Whether to parse commit messages

        Returns:
            Conversation dictionary
        """
        # User message is the PR request
        user_content = f"{pr_title}\n\n{pr_description}"

        # Assistant response includes commits and code changes
        assistant_parts = []

        for commit in commits:
            # Add commit message as explanation
            if parse_commits and commit.get('message'):
                # Extract meaningful message (skip merge commits)
                message = commit['message']
                if not message.lower().startswith('merge'):
                    assistant_parts.append({
                        'type': 'text',
                        'text': message
                    })

            # Add code changes
            for file_data in commit.get('files', []):
                if file_data.get('patch'):
                    # Determine language from filename
                    filename = file_data['filename']
                    language = PRFormatter._detect_language(filename)

                    # Format the changes
                    code_text = PRFormatter._format_file_changes(file_data)

                    assistant_parts.append({
                        'type': language,
                        'text': code_text
                    })

        # If only one text part, use string instead of list
        if len(assistant_parts) == 1 and assistant_parts[0]['type'] == 'text':
            assistant_content = assistant_parts[0]['text']
        elif assistant_parts:
            assistant_content = assistant_parts
        else:
            # Fallback to simple text
            assistant_content = "Code changes applied."

        return {
            'messages': [
                {'role': 'user', 'content': user_content},
                {'role': 'assistant', 'content': assistant_content}
            ]
        }

    @staticmethod
    def _detect_language(filename: str) -> str:
        """Detect programming language from filename.

        Args:
            filename: File name

        Returns:
            Language identifier
        """
        ext_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.jsx': 'javascript',
            '.tsx': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
            '.swift': 'swift',
            '.kt': 'kotlin',
            '.scala': 'scala',
            '.sh': 'bash',
            '.bash': 'bash',
        }

        ext = '.' + filename.split('.')[-1] if '.' in filename else ''
        return ext_map.get(ext.lower(), 'text')

    @staticmethod
    def _format_file_changes(file_data: Dict[str, Any]) -> str:
        """Format file changes into readable code.

        Args:
            file_data: File change data

        Returns:
            Formatted code string
        """
        filename = file_data['filename']
        patch = file_data.get('patch', '')

        output = [f"# {filename}"]

        if patch:
            # Extract just the changed lines (remove diff markers for cleaner code)
            lines = patch.split('\n')
            clean_lines = []

            for line in lines:
                # Skip diff headers
                if line.startswith('@@') or line.startswith('+++') or line.startswith('---'):
                    continue
                # Remove + and - markers but keep the content
                if line.startswith('+') and not line.startswith('+++'):
                    clean_lines.append(line[1:])
                elif line.startswith('-') and not line.startswith('---'):
                    # Skip deletions for now (focus on additions)
                    continue
                elif line.startswith(' '):
                    # Context line
                    clean_lines.append(line[1:])

            output.extend(clean_lines)

        return '\n'.join(output)
