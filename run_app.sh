echo "Starting Streamlit App..."
echo "If you are on a remote server, run this on your LOCAL machine to forward the port:"
echo "ssh -L 8501:localhost:8501 vavaghad@sg033.cs.washington.edu"
echo ""
echo "Then open: http://localhost:8501"
echo ""

streamlit run app.py --server.headless true --server.address 0.0.0.0 --server.port 8501
