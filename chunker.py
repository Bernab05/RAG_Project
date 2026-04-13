"""
Chunker structurel pour le Règlement EU 2024/1689 (AI Act).
Découpe le texte Markdown en chunks sémantiques par article, considérant et annexe,
avec métadonnées hiérarchiques (chapitre, section, article, titre).

Supporte 2 formats de fichier source :
- Ancien : CHAPITRE I / Article premier / |(1)|texte|
- Nouveau (PDF→MD) : ## CHAPITRE I / ## Article premier / (1) texte
"""

import re
from pathlib import Path

MD_FILE = Path(__file__).parent / "OJ_L_202401689_FR_TXTavec annexes.md"


def clean_text(text: str) -> str:
    """Nettoie les artefacts Markdown : séparateurs de tableau, liens, pages."""
    # Supprimer les lignes |---|---|
    text = re.sub(r"\|---\|---\|", "", text)
    # Supprimer le formatage tableau |a)| ... | -> a) ...
    text = re.sub(r"\|([a-z]\))\|(.+?)\|", r"\1 \2", text)
    # Supprimer les liens Markdown, garder le texte
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    # Supprimer les marqueurs de page PDF (ex: 44/144   ELI: http://...)
    text = re.sub(r"\d+/\d+\s+ELI:\s+http://\S+", "", text)
    # Supprimer les en-têtes de page JO
    text = re.sub(r"JO L du \d+\.\d+\.\d+\s+FR", "", text)
    # Supprimer les ## restants (titres markdown)
    text = re.sub(r"^##\s*", "", text, flags=re.MULTILINE)
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

    # ================================================================
    # Phase 1 : Considérants (avant CHAPITRE I)
    # ================================================================
    chapitre_start = None
    for idx, line in enumerate(lines):
        stripped = line.strip().replace("##", "").strip()
        if stripped == "CHAPITRE I":
            chapitre_start = idx
            break

    if chapitre_start:
        # Format 1 (ancien) : |(N)|texte|
        # Format 2 (nouveau) : (N)  texte multiligne
        current_num = None
        current_body = []

        def flush_considerant():
            if current_num and current_body:
                body = " ".join(current_body)
                cleaned = clean_text(body)
                if len(cleaned) >= 20:
                    chunks.append({
                        "content": f"Considérant {current_num} : {cleaned}",
                        "metadata": {
                            "type": "considerant",
                            "numero": current_num,
                            "chapter": "",
                            "section": "",
                            "article": "",
                            "title": f"Considérant {current_num}",
                        },
                    })

        for idx in range(chapitre_start):
            line = lines[idx]

            # Format ancien : |(N)|texte|
            m_old = re.match(r"^\|(\(\d+\))\|(.+)\|$", line)
            if m_old:
                flush_considerant()
                current_num = m_old.group(1)
                current_body = [m_old.group(2)]
                continue

            # Format nouveau : (N) texte
            m_new = re.match(r"^\((\d+)\)\s+(.+)", line.strip())
            if m_new:
                flush_considerant()
                current_num = f"({m_new.group(1)})"
                current_body = [m_new.group(2)]
                continue

            # Ligne de continuation (non vide, pas un header)
            stripped = line.strip()
            if stripped and current_num and not stripped.startswith("#"):
                current_body.append(stripped)

        flush_considerant()

    # ================================================================
    # Phase 2 : Articles (après CHAPITRE I, avant ANNEXE I)
    # ================================================================
    current_chapter = ""
    current_chapter_title = ""
    current_section = ""
    current_section_title = ""

    # Trouver la fin des articles (debut des annexes)
    annexe_start = len(lines)
    for idx, line in enumerate(lines):
        if re.match(r"^(##\s+)?ANNEXE\s+[IVXLC]+\s*$", line.strip()):
            annexe_start = idx
            break

    i = chapitre_start or 0
    while i < annexe_start:
        line = lines[i].strip()
        # Retirer le ## pour matcher uniformément
        bare = line.replace("##", "").strip()

        # Détecter un chapitre
        match_chap = re.match(r"^CHAPITRE\s+([IVXLC]+)$", bare)
        if match_chap:
            current_chapter = match_chap.group(1)
            current_section = ""
            current_section_title = ""
            j = i + 1
            while j < annexe_start and lines[j].strip().replace("##", "").strip() == "":
                j += 1
            if j < annexe_start:
                current_chapter_title = lines[j].strip().replace("##", "").strip()
            i = j + 1
            continue

        # Détecter une section
        match_sec = re.match(r"^SECTION\s+(\d+)$", bare)
        if match_sec:
            current_section = match_sec.group(1)
            j = i + 1
            while j < annexe_start and lines[j].strip().replace("##", "").strip() == "":
                j += 1
            if j < annexe_start:
                current_section_title = lines[j].strip().replace("##", "").strip()
            i = j + 1
            continue

        # Détecter un article
        match_art = re.match(r"^Article\s+(premier|\d+)$", bare)
        if match_art:
            art_num = match_art.group(1)
            if art_num == "premier":
                art_num = "1"

            # Titre de l'article = ligne suivante non vide
            j = i + 1
            while j < annexe_start and lines[j].strip().replace("##", "").strip() == "":
                j += 1
            art_title = lines[j].strip().replace("##", "").strip() if j < annexe_start else ""
            j += 1

            # Collecter le contenu jusqu'au prochain Article, CHAPITRE, SECTION
            content_lines = []
            while j < annexe_start:
                next_bare = lines[j].strip().replace("##", "").strip()
                if re.match(r"^(Article\s+(premier|\d+)|CHAPITRE\s+[IVXLC]+|SECTION\s+\d+)$", next_bare):
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
                    chunks.append({
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
                    })
            else:
                chunks.append({
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
                })

            i = j
            continue

        i += 1

    # ================================================================
    # Phase 3 : Annexes (après les articles)
    # Format : ## ANNEXE I  /  ## Titre de l'annexe  /  contenu
    # ================================================================
    i = annexe_start
    while i < len(lines):
        line = lines[i].strip()
        bare = line.replace("##", "").strip()

        match_annexe = re.match(r"^ANNEXE\s+([IVXLC]+)$", bare)
        if match_annexe:
            annexe_num = match_annexe.group(1)

            # Titre de l'annexe = ligne suivante non vide
            j = i + 1
            while j < len(lines) and lines[j].strip().replace("##", "").strip() == "":
                j += 1
            annexe_title = lines[j].strip().replace("##", "").strip() if j < len(lines) else ""
            j += 1

            # Collecter le contenu jusqu'à la prochaine ANNEXE ou fin de fichier
            content_lines = []
            while j < len(lines):
                next_bare = lines[j].strip().replace("##", "").strip()
                if re.match(r"^ANNEXE\s+[IVXLC]+$", next_bare):
                    break
                content_lines.append(lines[j])
                j += 1

            raw_content = "\n".join(content_lines)
            cleaned = clean_text(raw_content)

            if len(cleaned) < 10:
                i = j
                continue

            prefix = f"Annexe {annexe_num} : {annexe_title}"
            full_content = f"{prefix}\n\n{cleaned}"

            # Découper les annexes longues en chunks de ~2000 caractères
            if len(full_content) > 2000:
                # Découper par sections numérotées ou tirets
                sections = re.split(r"\n(?=\d+\.\s{2,}|\d+\.\s+[A-Z])", cleaned)
                for s_idx, section in enumerate(sections):
                    section = section.strip()
                    if len(section) < 20:
                        continue
                    chunks.append({
                        "content": f"{prefix}\n\n{section}",
                        "metadata": {
                            "type": "annexe",
                            "annexe": annexe_num,
                            "title": annexe_title,
                            "section": str(s_idx + 1),
                            "chapter": "",
                            "article": "",
                        },
                    })
            else:
                chunks.append({
                    "content": full_content,
                    "metadata": {
                        "type": "annexe",
                        "annexe": annexe_num,
                        "title": annexe_title,
                        "section": "",
                        "chapter": "",
                        "article": "",
                    },
                })

            i = j
            continue

        i += 1

    return chunks


if __name__ == "__main__":
    chunks = parse_ai_act()
    print(f"Nombre total de chunks : {len(chunks)}")
    considerants = [c for c in chunks if c["metadata"]["type"] == "considerant"]
    articles = [c for c in chunks if c["metadata"]["type"] == "article"]
    annexes = [c for c in chunks if c["metadata"]["type"] == "annexe"]
    print(f"  - Considérants : {len(considerants)}")
    print(f"  - Articles     : {len(articles)}")
    print(f"  - Annexes      : {len(annexes)}")
    print()
    # Afficher les annexes trouvées
    annexe_nums = sorted(set(c["metadata"]["annexe"] for c in annexes),
                         key=lambda x: len(x) * 100 + ord(x[0]))
    print(f"Annexes trouvées : {', '.join(annexe_nums)}")
    print()
    # Quelques exemples
    for c in chunks[:3] + [c for c in chunks if c["metadata"]["type"] == "annexe"][:3]:
        print(f"[{c['metadata']['type']}] {c['metadata'].get('title', '')}")
        print(c["content"][:200])
        print("---")
