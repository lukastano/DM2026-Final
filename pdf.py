from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib import colors

def create_progress_presentation():
    """
    Create a PDF presentation for the progress check
    """
    
    # Create PDF
    pdf_filename = "progress_check_presentation.pdf"
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    
    # Container for content
    story = []
    
    # Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=28,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=30,
        alignment=TA_CENTER,
        fontName='Helvetica-Bold'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=20,
        textColor=colors.HexColor('#1f4788'),
        spaceAfter=12,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=16,
        textColor=colors.HexColor('#2c5aa0'),
        spaceAfter=8,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=12,
        spaceAfter=6,
        leftIndent=20
    )
    
    # ==================== SLIDE 1: TITLE ====================
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Progress Check Presentation", title_style))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("Natural Disaster Severity Prediction", heading_style))
    story.append(Spacer(1, 0.5*inch))
    story.append(Paragraph("Team [Your Group ID]", subheading_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph("Data Mining Final Project", body_style))
    story.append(Paragraph("NYCU - Spring 2026", body_style))
    story.append(Paragraph("May 21, 2026", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 2: PROJECT OVERVIEW ====================
    story.append(Paragraph("Project Overview", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Goal</b>", subheading_style))
    story.append(Paragraph("• Predict drought severity (0-5 scale) for next 5 weeks", body_style))
    story.append(Paragraph("• Using only historical meteorological data", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Dataset</b>", subheading_style))
    story.append(Paragraph("• Training: 2,248 regions × 5,480 days = 12.3M rows", body_style))
    story.append(Paragraph("• Testing: 2,248 regions × 91 days", body_style))
    story.append(Paragraph("• Features: 14 weather variables", body_style))
    story.append(Paragraph("  - Temperature (tmp, tmp_max, tmp_min, tmp_range)", body_style))
    story.append(Paragraph("  - Humidity, Precipitation, Wind, Pressure", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 3: APPROACH ====================
    story.append(Paragraph("Current Approach", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Model Architecture</b>", subheading_style))
    story.append(Paragraph("• Random Forest Regressor", body_style))
    story.append(Paragraph("  - 200 trees, max depth 20", body_style))
    story.append(Paragraph("  - Trained on 1.76M weekly drought scores", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Features Used (14)</b>", subheading_style))
    story.append(Paragraph("• Wind: speed, min, max, range", body_style))
    story.append(Paragraph("• Temperature: tmp, max, min, range, surface", body_style))
    story.append(Paragraph("• Humidity, Precipitation", body_style))
    story.append(Paragraph("• Pressure: surface, dew point, wet bulb", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>Prediction Strategy</b>", subheading_style))
    story.append(Paragraph("• Use last 28 days average of weather data", body_style))
    story.append(Paragraph("• Generate 5-week forecast with trend adjustment", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 4: RESULTS ====================
    story.append(Paragraph("Current Results", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Performance Metrics</b>", subheading_style))
    story.append(Paragraph("• Validation MAE: 0.8072", body_style))
    story.append(Paragraph("• Public Leaderboard MAE: 0.8973", body_style))
    story.append(Paragraph("• Current Rank: #6 out of 10 teams", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Results table
    results_data = [
        ['Baseline', 'Score', 'Status'],
        ['Baseline 3', '0.8056', '❌ Need to beat'],
        ['Baseline 2', '0.8623', '✅ Beat'],
        ['Baseline 1', '0.9117', '✅ Beat'],
        ['Our Model', '0.8973', 'Current']
    ]
    
    results_table = Table(results_data, colWidths=[2*inch, 1.5*inch, 2*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(results_table)
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph("<b>Gap to Close:</b> Need ~10% improvement to beat Baseline 3", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 5: DATA INSIGHTS ====================
    story.append(Paragraph("Data Exploration Findings", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Score Distribution (Highly Imbalanced)</b>", subheading_style))
    
    # Distribution table
    dist_data = [
        ['Score', 'Count', 'Percentage', 'Severity'],
        ['0', '1,048,333', '60%', 'No drought'],
        ['1', '303,432', '17%', 'Mild'],
        ['2', '186,279', '11%', 'Moderate'],
        ['3', '118,496', '7%', 'Severe'],
        ['4', '69,422', '4%', 'Very Severe'],
        ['5', '31,974', '2%', 'Extreme']
    ]
    
    dist_table = Table(dist_data, colWidths=[0.8*inch, 1.5*inch, 1.3*inch, 1.8*inch])
    dist_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f4788')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.lightblue),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    story.append(dist_table)
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Key Patterns</b>", subheading_style))
    story.append(Paragraph("• Scores recorded weekly (every 7 days)", body_style))
    story.append(Paragraph("• Most regions have low/no drought (60%)", body_style))
    story.append(Paragraph("• Extreme drought is rare (only 2%)", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 6: CHALLENGES ====================
    story.append(Paragraph("Challenges Encountered", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>1. Weekly Score Pattern</b>", subheading_style))
    story.append(Paragraph("• Scores only recorded once per 7 days", body_style))
    story.append(Paragraph("• Other 6 days are NaN → Need special handling", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>2. Multi-Week Prediction</b>", subheading_style))
    story.append(Paragraph("• Predicting 5 weeks ahead is harder than 1 week", body_style))
    story.append(Paragraph("• Uncertainty increases with time horizon", body_style))
    story.append(Paragraph("• No future weather data available", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>3. Model Performance Gap</b>", subheading_style))
    story.append(Paragraph("• Current score: 0.8973", body_style))
    story.append(Paragraph("• Target: < 0.8056 (need ~10% improvement)", body_style))
    story.append(Paragraph("• Simple features may not capture complex drought patterns", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>4. Data Imbalance</b>", subheading_style))
    story.append(Paragraph("• 60% of data has no drought (score 0)", body_style))
    story.append(Paragraph("• Model may be biased toward predicting low scores", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 7: NEXT STEPS ====================
    story.append(Paragraph("Next Steps & Improvements", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>1. Feature Engineering (High Priority)</b>", subheading_style))
    story.append(Paragraph("• Rolling averages (7, 14, 28 days)", body_style))
    story.append(Paragraph("• Lag features (previous weeks' data)", body_style))
    story.append(Paragraph("• Drought indicators:", body_style))
    story.append(Paragraph("  - Consecutive dry days", body_style))
    story.append(Paragraph("  - Heat-dryness index", body_style))
    story.append(Paragraph("  - Moisture deficit", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>2. Advanced Models</b>", subheading_style))
    story.append(Paragraph("• XGBoost / LightGBM (gradient boosting)", body_style))
    story.append(Paragraph("• Model ensemble (combine multiple models)", body_style))
    story.append(Paragraph("• Hyperparameter tuning", body_style))
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("<b>3. Better Prediction Strategy</b>", subheading_style))
    story.append(Paragraph("• Trend analysis (is drought worsening?)", body_style))
    story.append(Paragraph("• Region-specific patterns", body_style))
    story.append(Paragraph("• Temporal feature importance", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>Timeline:</b>", subheading_style))
    story.append(Paragraph("• Next submission: This week", body_style))
    story.append(Paragraph("• Target: Beat Baseline 3 by June 10", body_style))
    story.append(PageBreak())
    
    # ==================== SLIDE 8: SUMMARY ====================
    story.append(Paragraph("Summary", heading_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>What We've Done</b>", subheading_style))
    story.append(Paragraph("✅ Downloaded and explored data", body_style))
    story.append(Paragraph("✅ Built baseline Random Forest model", body_style))
    story.append(Paragraph("✅ Successfully submitted to Kaggle (0.8973 MAE)", body_style))
    story.append(Paragraph("✅ Ranked #6 out of 10 teams", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>What's Working</b>", subheading_style))
    story.append(Paragraph("✅ Model successfully predicts general drought levels", body_style))
    story.append(Paragraph("✅ Beat 2 out of 3 baselines", body_style))
    story.append(Paragraph("✅ Solid foundation for improvement", body_style))
    story.append(Spacer(1, 0.3*inch))
    
    story.append(Paragraph("<b>What Needs Improvement</b>", subheading_style))
    story.append(Paragraph("❌ Need better features to capture drought patterns", body_style))
    story.append(Paragraph("❌ Multi-week predictions need refinement", body_style))
    story.append(Paragraph("❌ Must beat Baseline 3 (0.8056)", body_style))
    story.append(Spacer(1, 0.5*inch))
    
    story.append(Paragraph("<b>Confidence Level:</b> Moderate", subheading_style))
    story.append(Paragraph("We have a solid foundation and clear path to improvement.", body_style))
    
    # Build PDF
    doc.build(story)
    print(f"✓ Presentation created: {pdf_filename}")

if __name__ == "__main__":
    create_progress_presentation()