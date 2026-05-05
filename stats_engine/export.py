import os
import html
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime


class ReportExporter:
    """Export robuste des résultats statistiques en TXT, PDF, DOCX, ODT.

    Améliorations v2 :
    - PDF via ReportLab en priorité pour meilleure compatibilité Unicode ;
    - échappement HTML/XML pour les Paragraph ReportLab ;
    - chemins gérés avec pathlib ;
    - fallback TXT explicite avec chemin robuste ;
    - API compatible : add_section(), add_summary(), export_*(), export().
    """

    def __init__(self, report_title: str = "Rapport statistique"):
        self.title = str(report_title or "Rapport statistique")
        self.sections: List[Dict[str, Any]] = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_section(self, title: str, content: str):
        self.sections.append({"title": str(title or "Sans titre"), "content": "" if content is None else str(content)})

    def add_summary(self, summary: str):
        self.add_section("RÉSUMÉ", summary)

    def _ensure_parent(self, filepath: str) -> Path:
        path = Path(filepath)
        if path.parent and str(path.parent) not in ("", "."):
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _fallback_txt_path(self, filepath: str) -> Path:
        path = Path(filepath)
        if path.suffix:
            return path.with_suffix(".txt")
        return path.parent / f"{path.name}.txt"

    def export_txt(self, filepath: str):
        path = self._ensure_parent(filepath)
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"{'=' * 60}\n")
            f.write(f"{self.title}\n")
            f.write(f"Date: {self.timestamp}\n")
            f.write(f"{'=' * 60}\n\n")
            for section in self.sections:
                f.write(f"\n{'─' * 60}\n")
                f.write(f"  {section['title']}\n")
                f.write(f"{'─' * 60}\n\n")
                f.write(section["content"])
                f.write("\n\n")
            f.write(f"\n{'=' * 60}\n")
            f.write("Fin du rapport\n")
        return str(path)

    def export_pdf(self, filepath: str):
        """Export PDF via ReportLab. En cas d'absence de ReportLab, crée un TXT de fallback."""
        path = self._ensure_parent(filepath)
        try:
            self._export_pdf_reportlab(str(path))
            return str(path)
        except ImportError as exc:
            txt_path = self._fallback_txt_path(str(path))
            self.export_txt(str(txt_path))
            raise ImportError(f"ReportLab non disponible. Export TXT créé à la place : {txt_path}") from exc

    def _export_pdf_reportlab(self, filepath: str):
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.pdfbase.pdfmetrics import stringWidth

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=1.5 * cm,
            leftMargin=1.5 * cm,
            topMargin=1.5 * cm,
            bottomMargin=1.5 * cm,
        )
        styles = getSampleStyleSheet()
        code_style = ParagraphStyle(
            "StatProCode",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8,
            leading=10,
            wordWrap="CJK",
        )
        story = []
        story.append(Paragraph(f"<b><font size='18'>{html.escape(self.title)}</font></b>", styles["Title"]))
        story.append(Spacer(1, 8))
        story.append(Paragraph(f"Date: {html.escape(self.timestamp)}", styles["Normal"]))
        story.append(Spacer(1, 18))

        for idx, section in enumerate(self.sections):
            if idx > 0:
                story.append(PageBreak())
            story.append(Paragraph(f"<b>{html.escape(section['title'])}</b>", styles["Heading2"]))
            story.append(Spacer(1, 8))
            for block in section["content"].split("\n"):
                if block.strip() == "":
                    story.append(Spacer(1, 4))
                else:
                    # Preformatted préserve les alignements de tableaux texte et gère correctement les caractères échappés.
                    story.append(Preformatted(html.escape(block), code_style))
        doc.build(story)

    def export_docx(self, filepath: str):
        path = self._ensure_parent(filepath)
        try:
            from docx import Document
            from docx.shared import Pt
            doc = Document()
            doc.add_heading(self.title, level=0)
            doc.add_paragraph(f"Date: {self.timestamp}")
            doc.add_paragraph("")
            for section in self.sections:
                doc.add_heading(section["title"], level=1)
                for line in section["content"].split("\n"):
                    p = doc.add_paragraph(line)
                    p.paragraph_format.space_after = Pt(2)
            doc.save(str(path))
            return str(path)
        except ImportError as exc:
            txt_path = self._fallback_txt_path(str(path))
            self.export_txt(str(txt_path))
            raise ImportError(f"python-docx non disponible. Export TXT créé à la place : {txt_path}") from exc

    def export_odt(self, filepath: str):
        path = self._ensure_parent(filepath)
        try:
            from odf.opendocument import OpenDocumentText
            from odf.text import P, H
            doc = OpenDocumentText()
            doc.text.addElement(P(text=self.title))
            doc.text.addElement(P(text=f"Date: {self.timestamp}"))
            for section in self.sections:
                doc.text.addElement(H(text=section["title"], outlinelevel=1))
                for line in section["content"].split("\n"):
                    if line.strip():
                        doc.text.addElement(P(text=line))
            doc.save(str(path))
            return str(path)
        except ImportError as exc:
            txt_path = self._fallback_txt_path(str(path))
            self.export_txt(str(txt_path))
            raise ImportError(f"odfpy non disponible. Export TXT créé à la place : {txt_path}") from exc

    def export(self, filepath: str):
        ext = Path(filepath).suffix.lower()
        if ext == ".txt" or ext == "":
            return self.export_txt(filepath)
        if ext == ".pdf":
            return self.export_pdf(filepath)
        if ext == ".docx":
            return self.export_docx(filepath)
        if ext == ".odt":
            return self.export_odt(filepath)
        # Extension inconnue : conserve le chemin demandé mais écrit un contenu texte UTF-8.
        return self.export_txt(filepath)
