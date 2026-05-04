import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class ReportExporter:
    """Export des résultats statistiques en TXT, PDF, DOCX, ODT"""

    def __init__(self, report_title: str = "Rapport statistique"):
        self.title = report_title
        self.sections: List[Dict[str, Any]] = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def add_section(self, title: str, content: str):
        """Ajoute une section au rapport"""
        self.sections.append({"title": title, "content": content})

    def add_summary(self, summary: str):
        """Ajoute un résumé global"""
        self.add_section("RÉSUMÉ", summary)

    def export_txt(self, filepath: str):
        """Export en format texte brut"""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"{'='*60}\n")
            f.write(f"{self.title}\n")
            f.write(f"Date: {self.timestamp}\n")
            f.write(f"{'='*60}\n\n")

            for section in self.sections:
                f.write(f"\n{'─'*60}\n")
                f.write(f"  {section['title']}\n")
                f.write(f"{'─'*60}\n\n")
                f.write(section['content'])
                f.write("\n\n")

            f.write(f"\n{'='*60}\n")
            f.write(f"Fin du rapport\n")

    def export_pdf(self, filepath: str):
        """Export en PDF (via fpdf si disponible, sinon fallback TXT)"""
        try:
            from fpdf import FPDF
            self._export_pdf_fpdf(filepath)
        except ImportError:
            try:
                from reportlab.lib.pagesizes import letter
                from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                from reportlab.lib.styles import getSampleStyleSheet
                from reportlab.lib.units import inch
                self._export_pdf_reportlab(filepath)
            except ImportError:
                # Fallback: créer un TXT avec extension .pdf
                txt_path = filepath.replace('.pdf', '.txt')
                self.export_txt(txt_path)
                raise ImportError("Aucune bibliothèque PDF disponible. Export TXT créé à la place.")

    def _export_pdf_fpdf(self, filepath: str):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)

        # Page de titre
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 24)
        pdf.cell(0, 20, self.title, new_x="LM", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, f"Date: {self.timestamp}", new_x="LM", new_y="NEXT", align="C")
        pdf.ln(10)

        # Sections
        for section in self.sections:
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, section['title'], new_x="LM", new_y="NEXT")
            pdf.ln(5)

            pdf.set_font("Courier", "", 9)
            for line in section['content'].split('\n'):
                # Encodage sécurisé pour fpdf
                try:
                    pdf.cell(0, 5, line.encode('latin-1', 'replace').decode('latin-1'), new_x="LM", new_y="NEXT")
                except Exception:
                    pdf.cell(0, 5, line, new_x="LM", new_y="NEXT")
                pdf.ln(1)

        pdf.output(filepath)

    def _export_pdf_reportlab(self, filepath: str):
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(filepath, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []

        # Titre
        story.append(Paragraph(f"<b><font size=18>{self.title}</font></b>", styles['Title']))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"Date: {self.timestamp}", styles['Normal']))
        story.append(Spacer(1, 20))

        for section in self.sections:
            story.append(Paragraph(f"<b>{section['title']}</b>", styles['Heading2']))
            story.append(Spacer(1, 10))
            for line in section['content'].split('\n'):
                story.append(Paragraph(line, styles['Code']))
            story.append(PageBreak())

        doc.build(story)

    def export_docx(self, filepath: str):
        """Export en DOCX (via python-docx)"""
        try:
            from docx import Document
            from docx.shared import Pt, Inches
            doc = Document()

            # Titre
            doc.add_heading(self.title, level=0)
            doc.add_paragraph(f"Date: {self.timestamp}")
            doc.add_paragraph("")

            for section in self.sections:
                doc.add_heading(section['title'], level=1)
                doc.add_paragraph("")
                # Ajouter le contenu ligne par ligne
                for line in section['content'].split('\n'):
                    if line.strip():
                        p = doc.add_paragraph(line)
                        p.paragraph_format.space_after = Pt(2)

            doc.save(filepath)
        except ImportError:
            # Fallback: créer un TXT avec extension .docx
            txt_path = filepath.replace('.docx', '.txt')
            self.export_txt(txt_path)
            raise ImportError("python-docx non disponible. Export TXT créé à la place.")

    def export_odt(self, filepath: str):
        """Export en ODT (via odfpy)"""
        try:
            from odf.opendocument import OpenDocumentText
            from odf.text import P, H
            from odf.style import Style, TextProperties
            doc = OpenDocumentText()

            # Titre
            title = P(text=self.title)
            doc.text.addElement(title)

            date_p = P(text=f"Date: {self.timestamp}")
            doc.text.addElement(date_p)

            for section in self.sections:
                heading = H(text=section['title'], outlinelevel=1)
                doc.text.addElement(heading)
                for line in section['content'].split('\n'):
                    if line.strip():
                        p = P(text=line)
                        doc.text.addElement(p)

            doc.save(filepath)
        except ImportError:
            # Fallback: créer un TXT avec extension .odt
            txt_path = filepath.replace('.odt', '.txt')
            self.export_txt(txt_path)
            raise ImportError("odfpy non disponible. Export TXT créé à la place.")

    def export(self, filepath: str):
        """Export automatique selon l'extension"""
        ext = os.path.splitext(filepath)[1].lower()
        if ext == '.txt':
            self.export_txt(filepath)
        elif ext == '.pdf':
            self.export_pdf(filepath)
        elif ext == '.docx':
            self.export_docx(filepath)
        elif ext == '.odt':
            self.export_odt(filepath)
        else:
            self.export_txt(filepath)
