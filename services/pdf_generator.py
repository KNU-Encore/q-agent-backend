import markdown2
from weasyprint import HTML


def create_pdf_report(report_data, session_id):
    performance_report_md = report_data.get('performance_report', '')
    query_rewrite_md = report_data.get('query_rewrite', '')
    metrics = report_data.get('metrics', {})
    hallucination_report_data = report_data.get('hallucination_report', {})
    reliability_score = hallucination_report_data.get('reliability_score', 'N/A')
    confidence_level = hallucination_report_data.get('confidence_level', 'N/A')
    total_claims = hallucination_report_data.get('total_claims', 'N/A')
    verified_claims = hallucination_report_data.get('verified_claims', 'N/A')
    full_response_md = hallucination_report_data.get('full_response', '')
    verified_report_md = report_data.get('verified_report', '')

    extras = ["fenced-code-blocks", "tables"]
    performance_report_html = markdown2.markdown(performance_report_md, extras=extras)
    query_rewrite_html = markdown2.markdown(query_rewrite_md, extras=extras)
    full_response_html = markdown2.markdown(full_response_md, extras=extras)
    verified_report_html = markdown2.markdown(verified_report_md, extras=extras)

    metrics_html = "<h3>Performance Metrics</h3><table border='1' style='border-collapse: collapse; width: 50%;'>"
    for key, value in metrics.items():
        metrics_html += f"<tr><td style='padding: 5px;'>{key}</td><td style='padding: 5px;'>{value}</td></tr>"
    metrics_html += "</table>"

    hallucination_metrics_html = f"""
            <h3>Hallucination Metrics</h3>
            <table border='1' style='border-collapse: collapse; width: 50%;'>
                <tr><td style='padding: 5px;'>Reliability Score</td><td style='padding: 5px;'>{reliability_score}</td></tr>
                <tr><td style='padding: 5px;'>Confidence Level</td><td style='padding: 5px;'>{confidence_level}</td></tr>
                <tr><td style='padding: 5px;'>Total Claims</td><td style='padding: 5px;'>{total_claims}</td></tr>
                <tr><td style='padding: 5px;'>Verified Claims</td><td style='padding: 5px;'>{verified_claims}</td></tr>
            </table>
        """

    hallucination_report_html = f"""
            <h2>Hallucination Report</h2>
            {hallucination_metrics_html}
            <h3>Full Response</h3>
            <div>{full_response_html}</div>
            <h3>Verified Report</h3>
            <div>{verified_report_html}</div>
        """

    css_style = """
        body { font-family: 'Roboto', sans-serif; line-height: 1.6; }
        h1, h2, h3 { color: #333; border-bottom: 1px solid #eee; padding-bottom: 5px; }
        h1 { font-size: 24px; text-align: center; margin-bottom: 30px;}
        h2 { font-size: 20px; }
        h3 { font-size: 16px; }
        code {
            background-color: #f4f4f4; padding: 2px 4px; border-radius: 4px;
            font-family: monospace; color: #c7254e;
        }
        pre {
            background-color: #2d2d2d; color: #f8f8f2; padding: 15px;
            border-radius: 5px; white-space: pre-wrap; word-wrap: break-word;
        }
        table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        hr {
            border: 0; height: 1px;
            background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
            margin: 40px 0;
        }
        """

    full_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Slow Query Analysis Report</title>
            <style>{css_style}</style>
        </head>
        <body>
            <h1>Slow Query Analysis Report (Session ID: {session_id})</h1>
            <h2>Analysis Overview</h2>
            {performance_report_html}
            <hr>
            <h2>Query Tuning Guide</h2>
            {query_rewrite_html}
            <hr>
            {hallucination_report_html}
            <hr>
            {metrics_html}
        </body>
        </html>
        """

    return HTML(string=full_html).write_pdf()
