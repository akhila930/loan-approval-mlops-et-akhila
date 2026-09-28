const form = document.getElementById("loanForm");
const predictBtn = document.getElementById("predictBtn");

const resultSection = document.getElementById("result");
const errorSection = document.getElementById("error");

const loanStatus = document.getElementById("loanStatus");
const approvalProbability =
    document.getElementById("approvalProbability");

const rejectionProbability =
    document.getElementById("rejectionProbability");

const errorMessage =
    document.getElementById("errorMessage");


form.addEventListener("submit", async function (event) {

    event.preventDefault();

    resultSection.classList.add("hidden");
    errorSection.classList.add("hidden");

    predictBtn.disabled = true;
    predictBtn.textContent = "Running Prediction...";


    const data = {

        Gender:
            document.getElementById("Gender").value,

        Married:
            document.getElementById("Married").value,

        Dependents:
            Number(document.getElementById("Dependents").value),

        Education:
            document.getElementById("Education").value,

        Employment_Status:
            document.getElementById("Employment_Status").value,

        Applicant_Income:
            Number(document.getElementById("Applicant_Income").value),

        Coapplicant_Income:
            Number(document.getElementById("Coapplicant_Income").value),

        Loan_Amount:
            Number(document.getElementById("Loan_Amount").value),

        Loan_Term:
            Number(document.getElementById("Loan_Term").value),

        Credit_History:
            Number(document.getElementById("Credit_History").value),

        Property_Area:
            document.getElementById("Property_Area").value,

        Age:
            Number(document.getElementById("Age").value)
    };


    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify(data)
        });


        const result = await response.json();


        if (!response.ok) {
            throw new Error(
                result.detail || "Prediction request failed."
            );
        }


        loanStatus.textContent = result.loan_status;

        approvalProbability.textContent =
            `${(result.approval_probability * 100).toFixed(2)}%`;

        rejectionProbability.textContent =
            `${(result.rejection_probability * 100).toFixed(2)}%`;


        resultSection.classList.remove("hidden");


    } catch (error) {

        errorMessage.textContent = error.message;

        errorSection.classList.remove("hidden");

    } finally {

        predictBtn.disabled = false;
        predictBtn.textContent = "Predict Loan Approval";
    }

});
