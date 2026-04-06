# memory/skills_files.py
"""
Skill storage using directory-based SKILL.md files.
Each skill is stored as memory/skills/<intent-name>/SKILL.md
with YAML frontmatter (name, description, category, intent)
and markdown content body.
"""

import os
import re
from pathlib import Path
from typing import Dict, Optional
from utils.config import SKILLS_DIR


def _slugify(text: str) -> str:
    """Convert intent/name to a valid directory name."""
    return re.sub(r'[^a-z0-9]+', '-', text.lower()).strip('-')


def create_skill(
    name: str,
    description: str,
    content: str,
    category: str = "",
    intent: str = "",
) -> Path:
    """
    Create a skill directory with a SKILL.md file.
    name: slugified intent used as directory name
    description: one line summary of what the skill handles
    content: full skill instructions in markdown
    """
    base_dir = Path(SKILLS_DIR)
    skill_dir = base_dir / name
    skill_dir.mkdir(parents=True, exist_ok=True)

    # Escape description if it contains special YAML characters
    escaped_description = description
    if ':' in description or '"' in description or "'" in description:
        escaped_description = description.replace('"', '\\"')
        escaped_description = f'"{escaped_description}"'

    skill_content = f"""---
name: {name}
description: {escaped_description}
category: {category}
intent: {intent}
---

{content}"""

    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(skill_content, encoding='utf-8')
    print(f"  [Skills Files] Skill saved: {name}")
    return skill_file


def read_skill(name: str) -> Dict[str, str]:
    """
    Read a skill from its SKILL.md file by name.
    Returns dict with keys: name, description, category, intent, content
    """
    base_dir = Path(SKILLS_DIR)
    skill_file = base_dir / name / "SKILL.md"

    if not skill_file.exists():
        raise FileNotFoundError(f"SKILL.md not found for skill '{name}'")

    file_content = skill_file.read_text(encoding='utf-8')

    pattern = r'^---\s*\n(.*?)\n---\s*\n(.*)$'
    match = re.match(pattern, file_content, re.DOTALL)

    if not match:
        raise ValueError(f"Invalid SKILL.md format for skill '{name}'")

    frontmatter = match.group(1)
    content = match.group(2).strip()

    def extract_field(field: str) -> str:
        m = re.search(rf'^{field}:\s*(.+)$', frontmatter, re.MULTILINE)
        if not m:
            return ""
        val = m.group(1).strip()
        if (val.startswith('"') and val.endswith('"')):
            val = val[1:-1].replace('\\"', '"')
        elif (val.startswith("'") and val.endswith("'")):
            val = val[1:-1].replace("\\'", "'")
        return val

    return {
        'name': extract_field('name'),
        'description': extract_field('description'),
        'category': extract_field('category'),
        'intent': extract_field('intent'),
        'content': content,
    }


def skill_exists(intent: str) -> bool:
    """Check if a skill already exists for this intent."""
    name = _slugify(intent)
    skill_file = Path(SKILLS_DIR) / name / "SKILL.md"
    return skill_file.exists()


def list_skills(base_path: Optional[str] = None) -> list:
    """List all skill directory names that contain a SKILL.md."""
    base_dir = Path(base_path) if base_path else Path(SKILLS_DIR)
    if not base_dir.exists():
        return []
    skills = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and (item / "SKILL.md").exists():
            skills.append(item.name)
    return skills


def list_skills_detailed(base_path: Optional[str] = None) -> list:
    """List all skills with name and description."""
    base_dir = Path(base_path) if base_path else Path(SKILLS_DIR)
    if not base_dir.exists():
        return []
    skills = []
    for item in sorted(base_dir.iterdir()):
        if item.is_dir() and (item / "SKILL.md").exists():
            try:
                skill_data = read_skill(item.name)
                skills.append({
                    'name': skill_data['name'],
                    'description': skill_data['description'],
                    'category': skill_data['category'],
                    'intent': skill_data['intent'],
                })
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Skipping {item.name} — {e}")
    return skills


def format_skills(skills: list) -> list:
    """Format skills list as XML-style tags for prompt injection."""
    formatted = []
    for skill in skills:
        formatted.append(
f"""<skill>
    <name>{skill['name']}</name>
    <description>{skill['description']}</description>
</skill>""")
    return formatted


def get_skills_summary_text() -> str:
    """
    Returns formatted skill summaries ready to inject into agent prompt.
    Uses XML tags from format_skills.
    """
    skills = list_skills_detailed()
    if not skills:
        return "No skills available yet."

    formatted = format_skills(skills)
    header = "Available skills (use the name to fetch full details):\n"
    return header + "\n".join(formatted)


def save_skill_from_interaction(
    description: str,
    content: str,
    category: str,
    intent: str,
    query: str,
) -> Optional[Path]:
    """
    Saves a skill from a training interaction.
    Skips if a skill for this intent already exists.
    """
    name = _slugify(intent)

    if skill_exists(intent):
        print(f"  [Skills Files] Skill for intent '{intent}' already exists, skipping.")
        return None

    return create_skill(
        name=name,
        description=description,
        content=content,
        category=category,
        intent=intent,
    )


def get_skill_detail(name: str) -> str:
    """
    Fetch the full content of a skill by name.
    Returns the markdown content ready to inject into agent context.
    """
    try:
        skill = read_skill(name)
        return skill['content']
    except FileNotFoundError:
        return f"Skill '{name}' not found."