from flask import Flask, render_template, jsonify, request
import processor

app = Flask(__name__)

@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())

@app.route('/chatbot', methods=["POST"])
def chatbotResponse():
    if request.method == 'POST':
        try:
            the_question = request.form['question']

            if the_question.strip():  # Check if the question is not empty
                response = processor.chatbot_response(the_question)
            else:
                response = "Please ask a valid question."

        except Exception as e:
            response = f"Error processing your request: {str(e)}"
        
        return jsonify({"response": response })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
