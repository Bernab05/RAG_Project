"""
Chunker structurel pour le Règlement EU 2024/1689 (AI Act).
Découpe le texte Markdown en chunks sémantiques par article et considérant,
avec métadonnées hiérarchiques (chapitre, section, article, titre).
"""

import re
from pathlib import Path

MD_FILE = Path(__file__).parent / "L-202401689FR.000101.fmx.xml.md"


def clean_text(text: str) -> str:
    """Nettoie les artefacts Markdown : séparateurs de tableau, liens, YAML."""
    # Supprimer les lignes |---|---|
    text = re.sub(r"\|---\|---\|", "", text)
    # Supprimer le formatage tableau |a)| ... | -> a) ...
    text = re.sub(r"\|([a-z]\))\|(.+?)\|", r"\1 \2", text)
    # Supprimer les liens Markdown, garder le texte
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Nettoyer les lignes vides multiples
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def parse_ai_act(filepath: str | Path = MD_FILE) -> list[dict]:
    """
    Parse le fichier Markdown de l'AI Act et retourne une liste de chunks.
    Chaque chunk = {"content": str, "metadata": dict}
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Supprimer le header YAML
    text = re.sub(r"^---.*?---", "", text, count=1, flags=re.DOTALL)
    # Supprimer le titre H1
    text = re.sub(r"^#\s+.*$", "", text, flags=re.MULTILINE)
    # Supprimer le bloc Excerpt
    text = re.sub(r"^>\s+.*$", "", text, flags=re.MULTILINE)

    # Normaliser les espaces insécables
    text = text.replace("\xa0", " ")

    lines = text.split("\n")
    chunks = []

    current_chapter = ""
    current_chapter_title = ""
    current_section = ""
    current_section_title = ""

    i = 0
    # --- Phase 1 : Considérants (avant CHAPITRE I) ---
    chapitre_start = None
    for idx, line in enumerate(lines):
        if line.strip() == "CHAPITRE I":
            chapitre_start = idx
            break

    if chapitre_start:
        # Extraire chaque considérant : |(N)|texte| (une seule ligne)
        for idx in range(chapitre_start):
            m = re.match(r"^\|(\(\d+\))\|(.+)\|$", lines[idx])
            if not m:
                continue
            num, body = m.group(1), m.group(2)
            cleaned = clean_text(body)
            if len(cleaned) < 20:
                continue
            chunks.append(
                {
                    "content": f"Considérant {num} : {cleaned}",
                    "metadata": {
                        "type": "considerant",
                        "numero": num,
                        "chapter": "",
                        "section": "",
                        "article": "",
                        "title": f"Considérant {num}",
                    },
                }
            )
        i = chapitre_start

    # --- Phase 2 : Articles (après CHAPITRE I) ---
    while i < len(lines):
        line = lines[i].strip()

        # Détecter un chapitre
        match_chap = re.match(r"^CHAPITRE\s+([IVXLC]+)$", line)
        if match_chap:
            current_chapter = match_chap.group(1)
            current_section = ""
            current_section_title = ""
            # Le titre du chapitre est sur la ligne suivante non vide
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                current_chapter_title = lines[j].strip()
            i = j + 1
            continue

        # Détecter une section
        match_sec = re.match(r"^SECTION\s+(\d+)$", line)
        if match_sec:
            current_section = match_sec.group(1)
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            if j < len(lines):
                current_section_title = lines[j].strip()
            i = j + 1
            continue

        # Détecter un article
        match_art = re.match(r"^Article\s+(premier|\d+)$", line)
        if match_art:
            art_num = match_art.group(1)
            if art_num == "premier":
                art_num = "1"

            # Titre de l'article = ligne suivante non vide
            j = i + 1
            while j < len(lines) and lines[j].strip() == "":
                j += 1
            art_title = lines[j].strip() if j < len(lines) else ""
            j += 1

            # Collecter le contenu jusqu'au prochain Article, CHAPITRE, ou SECTION
            content_lines = []
            while j < len(lines):
                next_line = lines[j].strip()
                if re.match(r"^(Article\s+(premier|\d+)|CHAPITRE\s+[IVXLC]+|SECTION\s+\d+)$", next_line):
                    break
                content_lines.append(lines[j])
                j += 1

            raw_content = "\n".join(content_lines)
            cleaned = clean_text(raw_content)

            if len(cleaned) < 10:
                i = j
                continue

            # Construire le préfixe hiérarchique
            prefix_parts = []
            if current_chapter:
                prefix_parts.append(f"Chapitre {current_chapter} - {current_chapter_title}")
            if current_section:
                prefix_parts.append(f"Section {current_section} - {current_section_title}")
            prefix_parts.append(f"Article {art_num} : {art_title}")
            prefix = " > ".join(prefix_parts)

            full_content = f"{prefix}\n\n{cleaned}"

            # Si le chunk est trop long (> 2000 caractères), découper par paragraphe
            if len(full_content) > 2000:
                paragraphs = re.split(r"\n(?=\d+\.\s{2,})", cleaned)
                for p_idx, para in enumerate(paragraphs):
                    para = para.strip()
                    if len(para) < 20:
                        continue
                    chunks.append(
                        {
                            "content": f"{prefix}\n\n{para}",
                            "metadata": {
                                "type": "article",
                                "chapter": current_chapter,
                                "chapter_title": current_chapter_title,
                                "section": current_section,
                                "section_title": current_section_title,
                                "article": art_num,
                                "title": art_title,
                                "paragraph": str(p_idx + 1),
                            },
                        }
                    )
            else:
                chunks.append(
                    {
                        "content": full_content,
                        "metadata": {
                            "type": "article",
                            "chapter": current_chapter,
                            "chapter_title": current_chapter_title,
                            "section": current_section,
                            "section_title": current_section_title,
                            "article": art_num,
                            "title": art_title,
                            "paragraph": "",
                        },
                    }
                )

            i = j
            continue

        i += 1

    return chunks


if __name__ == "__main__":
    chunks = parse_ai_act()
    print(f"Nombre total de chunks : {len(chunks)}")
    considerants = [c for c in chunks if c["metadata"]["type"] == "considerant"]
    articles = [c for c in chunks if c["metadata"]["type"] == "article"]
    print(f"  - Considérants : {len(considerants)}")
    print(f"  - Articles     : {len(articles)}")
    print()
    # Afficher quelques exemples
    for c in chunks[:3]:
        print(f"[{c['metadata']['type']}] {c['metadata']['title']}")
        print(c["content"][:200])
        print("---")
