from flask import Flask, render_template, request, send_from_directory
from dotenv import load_dotenv
import os
import ai_verifier
import logging
import asyncio
import json
import pdfkit

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG for detailed logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    logger.error(f"404 error: {str(e)}")
    return render_template('index.html', error="Resource not found. Please check the URL or prompt."), 404

@app.route('/favicon.ico')
def favicon():
    """Serve favicon to prevent 404 errors."""
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/', methods=['GET', 'POST'])
async def index():
    """Handle main page and form submission."""
    results = None
    prompt = None
    error = None
    try:
        if request.method == 'POST':
            prompt = request.form.get('prompt', '').strip()
            logger.debug(f"Received prompt: {prompt}")
            if not prompt:
                logger.warning("Empty prompt submitted")
                error = "Please enter a prompt."
            else:
                try:
                    # Run the async process_prompt function
                    results = await ai_verifier.process_prompt(prompt)
                    logger.debug(f"Results: {json.dumps(results, indent=2)}")
                    if 'error' in results:
                        error = results['error']
                except Exception as e:
                    logger.error(f"Error processing prompt: {str(e)}", exc_info=True)
                    error = f"Processing failed: {str(e)}"
        
        return render_template('index.html', prompt=prompt, results=results, error=error)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return render_template('index.html', error="An unexpected error occurred."), 500

@app.route('/report', methods=['POST'])
async def generate_report():
    """Generate a detailed report."""
    prompt = request.form.get('prompt', '').strip()
    color_scheme = request.form.get('color_scheme', 'default')
    sections = request.form.get('sections', 'prompt,summary,references,api,visual,insights,analysis').split(',')
    logger.debug(f"Generating report for prompt: {prompt}, color_scheme: {color_scheme}, sections: {sections}")

    try:
        results = await ai_verifier.process_prompt(prompt)
        if 'error' in results:
            error = results['error']
            return render_template('report.html', prompt=prompt, results=None, error=error, color_scheme=color_scheme, sections=sections)

        # Render the report template with customization
        report_html = render_template('report.html', prompt=prompt, results=results, color_scheme=color_scheme, sections=sections)

        # Generate PDF from HTML
        pdf_path = "report.pdf"  # You can customize the path
        pdfkit.from_string(report_html, pdf_path)

        # Return the PDF file
        return send_from_directory(os.getcwd(), pdf_path, as_attachment=True)

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}", exc_info=True)
        return render_template('report.html', prompt=prompt, results=None, error=f"Report generation failed: {str(e)}", color_scheme=color_scheme, sections=sections)

if __name__ == '__main__':
    config = {
        'host': 'localhost',
        'port': 8000,
        'use_reloader': True,
        'debug': True
    }
    
    try:
        import hypercorn.asyncio
        from hypercorn.config import Config
        
        hypercorn_config = Config()
        hypercorn_config.bind = [f"{config['host']}:{config['port']}"]
        hypercorn_config.use_reloader = config['use_reloader']
        
        asyncio.run(hypercorn.asyncio.serve(app, hypercorn_config))
    except ImportError:
        logger.error("Hypercorn not installed. Please install using: pip install hypercorn")
        exit(1)