"""
Seed script: Creates 10 HyperVerge subjective questions with reference answers
in course_id=2, cohort_id=1, linked to milestone_id=1.

Run from src/ directory:
  ..\.venv\Scripts\python.exe seed_demo.py
"""

import sqlite3
import json
import os
import sys

# Add parent to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db", "db.sqlite")

QUESTIONS = [
    {
        "title": "HyperVerge Company Overview",
        "question": "What is HyperVerge and what problem does it solve in the identity verification space?",
        "reference_answer": (
            "HyperVerge is an AI-first company specializing in identity verification and onboarding automation. "
            "Founded in 2014 and headquartered in Bangalore, India, HyperVerge uses deep learning and computer vision "
            "to solve the problem of manual, slow, and error-prone KYC (Know Your Customer) processes. Their platform "
            "automates document verification (ID cards, passports, utility bills), face matching, and liveness detection, "
            "reducing onboarding time from days to seconds. They serve 100+ enterprise clients across fintech, banking, "
            "insurance, and telecom industries in 40+ countries."
        ),
    },
    {
        "title": "Core AI Technology",
        "question": "Explain the core AI technologies that power HyperVerge's products. How do deep learning and computer vision work together in their platform?",
        "reference_answer": (
            "HyperVerge's platform is powered by three core AI technologies: (1) Computer Vision for document analysis - "
            "their OCR engine extracts text from ID documents, passports, and utility bills with high accuracy even on "
            "low-quality images. (2) Deep Learning models for face recognition - they use convolutional neural networks "
            "trained on millions of face images to match a selfie against an ID photo with 99.5%+ accuracy. (3) Liveness "
            "detection using AI to prevent spoofing - their models can distinguish between a real person and a photo/video "
            "replay attack. These technologies work together in a pipeline: document is captured -> OCR extracts data -> "
            "face on document is compared to live selfie -> liveness check ensures the person is real -> all results are "
            "returned in under 3 seconds."
        ),
    },
    {
        "title": "KYC and Fintech Use Case",
        "question": "How does HyperVerge's solution transform the KYC onboarding process in fintech companies? Describe the before and after.",
        "reference_answer": (
            "Before HyperVerge: Fintech companies relied on manual KYC processes where agents physically verified documents, "
            "taking 2-5 business days per customer. Drop-off rates were 40-60% because customers abandoned lengthy processes. "
            "Fraud detection was inconsistent and relied on human judgment. After HyperVerge: The entire KYC process happens "
            "in under 60 seconds. Customers upload their ID document and take a selfie through the app. HyperVerge's AI "
            "automatically extracts document data via OCR, verifies the document's authenticity, matches the face on the "
            "document to the selfie, performs liveness detection, and checks against watchlists. Drop-off rates decrease to "
            "under 10%, fraud detection accuracy exceeds 99%, and companies can onboard thousands of customers per day "
            "without manual intervention. This transforms KYC from a cost center into a competitive advantage."
        ),
    },
    {
        "title": "Face Recognition Architecture",
        "question": "Describe the technical architecture of HyperVerge's face recognition system. What makes it robust against different lighting, angles, and demographics?",
        "reference_answer": (
            "HyperVerge's face recognition system uses a multi-stage deep learning architecture: (1) Face Detection - a "
            "lightweight neural network (similar to MTCNN or RetinaFace) locates faces in the image and outputs bounding "
            "boxes. (2) Face Alignment - detected faces are normalized for pose, rotation, and scale using facial landmark "
            "detection (eyes, nose, mouth corners). (3) Feature Extraction - a deep CNN (based on architectures like "
            "ArcFace or FaceNet) maps the aligned face into a 128/512-dimensional embedding vector. (4) Matching - cosine "
            "similarity between the selfie embedding and document face embedding determines if they're the same person. "
            "Robustness comes from: training on diverse datasets spanning multiple demographics, skin tones, and age groups; "
            "data augmentation with varying lighting, angles, occlusions (glasses, masks); and continuous model retraining "
            "on edge cases from production data. They achieve 99.5%+ accuracy across all demographics, which is critical "
            "for regulatory compliance in global markets."
        ),
    },
    {
        "title": "Liveness Detection",
        "question": "What is liveness detection and why is it critical for HyperVerge's identity verification? Explain the different types of spoofing attacks it prevents.",
        "reference_answer": (
            "Liveness detection is the AI's ability to determine whether the person in front of the camera is a real, "
            "live human being versus a spoofing attempt. It's critical because without it, fraudsters could use a printed "
            "photo, a video replay, or a 3D mask to impersonate someone during KYC verification. HyperVerge's liveness "
            "detection prevents several attack types: (1) Print attacks - holding up a printed photo of the victim. The AI "
            "detects lack of 3D depth, unnatural skin texture, and paper edges. (2) Screen replay attacks - showing a video "
            "of the victim on a phone/tablet. The AI detects moire patterns, screen reflections, and lack of natural micro-"
            "movements. (3) 3D mask attacks - wearing a realistic mask. The AI uses texture analysis and sometimes infrared "
            "to detect artificial materials. HyperVerge uses both passive liveness (single frame analysis, no user action "
            "required) and active liveness (asking user to blink, turn head, smile) depending on the security level required. "
            "Their passive liveness is preferred for UX because it requires zero user effort while still achieving 99%+ "
            "spoofing detection rate."
        ),
    },
    {
        "title": "Document OCR Pipeline",
        "question": "How does HyperVerge's document OCR pipeline work? What challenges does it face with real-world document images and how does it overcome them?",
        "reference_answer": (
            "HyperVerge's OCR pipeline processes identity documents through several stages: (1) Document Detection - "
            "AI identifies and crops the document from the camera frame, handling tilted or partially visible documents. "
            "(2) Document Classification - the system automatically identifies the document type (passport, Aadhaar, "
            "driver's license, etc.) from 100+ supported document types across 200+ countries. (3) Pre-processing - "
            "image enhancement for blurry, low-light, or glare-affected images using denoising and contrast adjustment. "
            "(4) Text Extraction - a combination of traditional OCR and deep learning-based text recognition extracts "
            "fields like name, date of birth, document number, and address. (5) Field Parsing - extracted text is "
            "structured into specific fields using NLP and regex patterns. Real-world challenges include: poor image "
            "quality (blurry photos taken by shaky hands), glare from laminated documents, partial occlusion (fingers "
            "covering corners), diverse document formats across countries, and handwritten fields. HyperVerge overcomes "
            "these through training on millions of real-world document images, aggressive data augmentation, and "
            "fallback to multiple OCR engines when confidence is low."
        ),
    },
    {
        "title": "Insurance Industry Application",
        "question": "How can HyperVerge's AI technology be applied to the insurance industry beyond just KYC? Describe at least three specific use cases.",
        "reference_answer": (
            "HyperVerge's AI extends well beyond KYC in insurance: (1) Claims Automation - when a policyholder submits "
            "a vehicle damage claim, HyperVerge's computer vision can analyze photos of the damage, classify the severity "
            "(minor scratch vs. major dent vs. total loss), estimate repair costs, and flag potentially fraudulent claims "
            "where damage appears staged or inconsistent. This reduces claims processing from weeks to hours. "
            "(2) Underwriting Risk Assessment - AI analyzes documents submitted during policy application (medical records, "
            "property photos, financial documents) to automatically assess risk factors and recommend premium adjustments. "
            "For example, analyzing property photos to assess flood risk or structural condition. "
            "(3) Agent Verification - insurance companies use HyperVerge to verify the identity of field agents during "
            "onboarding and for each customer interaction, preventing agent fraud where imposters sell fake policies. "
            "The face recognition ensures the person selling the policy is actually the registered agent. "
            "Additional use cases include: policy renewal automation (verify customer identity for digital renewals), "
            "hospital network verification (ensure the hospital submitting a claim is legitimate), and beneficiary "
            "verification during claim payouts."
        ),
    },
    {
        "title": "Data Privacy and Compliance",
        "question": "How does HyperVerge handle data privacy and regulatory compliance across different countries? What are the key regulations they must comply with?",
        "reference_answer": (
            "HyperVerge operates across 40+ countries and must comply with multiple regulatory frameworks: "
            "(1) GDPR (Europe) - requires explicit consent for biometric data processing, right to erasure, data "
            "minimization, and data processing agreements. HyperVerge processes data only for the stated purpose and "
            "deletes biometric data after verification unless explicitly authorized to retain it. "
            "(2) India's DPDP Act (Digital Personal Data Protection) - governs how Indian citizen data is collected, "
            "stored, and processed. HyperVerge ensures data localization requirements are met. "
            "(3) SOC 2 Type II Certification - demonstrates that HyperVerge's systems maintain security, availability, "
            "and confidentiality of customer data through independently audited controls. "
            "(4) ISO 27001 - information security management system certification. "
            "Key practices: all biometric data is encrypted at rest and in transit; data is processed in-region when "
            "required (no cross-border transfer); customers can choose between cloud processing and on-premise deployment; "
            "audit logs track every data access; and biometric data retention policies are configurable per client. "
            "HyperVerge also provides consent management SDKs that help their clients collect proper user consent "
            "before processing biometric data."
        ),
    },
    {
        "title": "Competitive Landscape",
        "question": "Who are HyperVerge's main competitors in the identity verification market? What differentiates HyperVerge from them?",
        "reference_answer": (
            "HyperVerge competes in the identity verification (IDV) market against several players: "
            "(1) Jumio - US-based, one of the largest IDV providers. Strong in North America and Europe but expensive "
            "and less optimized for Asian document types. HyperVerge differentiates with better accuracy on Indian and "
            "Southeast Asian documents and more competitive pricing. "
            "(2) Onfido - UK-based, strong brand in European markets. Focuses on regulatory compliance. HyperVerge "
            "offers faster processing speed and better handling of low-quality images common in emerging markets. "
            "(3) IDnow - German company focused on European video identification. More compliance-focused but less "
            "flexible for high-volume automated processing. "
            "(4) Digilocker/Aadhaar-based solutions - India-specific government infrastructure. Free but limited to "
            "India only and requires Aadhaar number which not all use cases have. "
            "HyperVerge's key differentiators: (a) Superior accuracy on Asian and African documents where competitors "
            "struggle due to training data bias, (b) Ultra-fast processing (sub-3-second verification), (c) Flexible "
            "deployment (cloud, on-premise, hybrid), (d) Competitive pricing for high-volume emerging market use cases, "
            "(e) Passive liveness detection that requires zero user interaction, and (f) Deep customization for "
            "enterprise clients including custom document types and business rules."
        ),
    },
    {
        "title": "Future of AI in Identity Verification",
        "question": "What is the future of AI-powered identity verification? Discuss emerging trends like decentralized identity, deepfakes, and regulatory changes that will shape companies like HyperVerge.",
        "reference_answer": (
            "The future of AI-powered identity verification is shaped by several converging trends: "
            "(1) Deepfake Threats - as generative AI makes it easy to create realistic fake videos and images, "
            "liveness detection must evolve. Next-gen systems will use multi-modal verification (voice + face + "
            "behavioral biometrics) and AI-powered deepfake detection models trained on synthetic media. Companies "
            "like HyperVerge will need continuous model updates to stay ahead of improving deepfake technology. "
            "(2) Decentralized Identity (DID) - blockchain-based verifiable credentials allow users to prove their "
            "identity without sharing raw documents. Users get a cryptographic proof of verification that can be "
            "reused across services. HyperVerge could become an 'issuer' of verified credentials rather than "
            "re-verifying from scratch each time. "
            "(3) Regulatory Tightening - the EU AI Act classifies biometric identification as 'high-risk AI,' "
            "requiring explainability, human oversight, and bias auditing. India's DPDP Act adds consent requirements. "
            "Compliance will become a product feature, not just a checkbox. "
            "(4) Continuous Authentication - moving from one-time KYC to ongoing identity verification throughout "
            "the customer relationship using behavioral biometrics (typing patterns, device usage patterns). "
            "(5) Edge AI - running verification models directly on mobile devices for privacy (biometric data never "
            "leaves the phone) and speed. HyperVerge's SDK architecture positions them well for this shift. "
            "The companies that thrive will combine robust anti-deepfake AI, privacy-preserving architectures, "
            "and regulatory expertise into a single platform."
        ),
    },
]


def make_block(text: str) -> list:
    """Create a BlockNote-compatible content block."""
    return [
        {
            "id": f"block-{hash(text) % 100000}",
            "type": "paragraph",
            "props": {
                "textColor": "default",
                "backgroundColor": "default",
                "textAlignment": "left",
            },
            "content": [{"type": "text", "text": text, "styles": {}}],
            "children": [],
        }
    ]


def seed():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    org_id = 1
    course_id = 2
    milestone_id = 1

    # Create the quiz task
    c.execute(
        """INSERT INTO tasks (org_id, type, title, status)
           VALUES (?, 'quiz', 'HyperVerge AI Assessment', 'published')""",
        (org_id,),
    )
    task_id = c.lastrowid
    print(f"Created task id={task_id}")

    # Link task to course + milestone
    # Get next ordering
    c.execute(
        "SELECT COALESCE(MAX(ordering), -1) + 1 FROM course_tasks WHERE course_id=? AND deleted_at IS NULL",
        (course_id,),
    )
    next_order = c.fetchone()[0]

    c.execute(
        """INSERT INTO course_tasks (task_id, course_id, ordering, milestone_id)
           VALUES (?, ?, ?, ?)""",
        (task_id, course_id, next_order, milestone_id),
    )
    print(f"Linked task to course={course_id}, milestone={milestone_id}, ordering={next_order}")

    # Insert 10 questions with reference answers stored in the 'answer' column
    for i, q in enumerate(QUESTIONS):
        blocks_json = json.dumps(make_block(q["question"]))
        answer_json = json.dumps(make_block(q["reference_answer"]))

        c.execute(
            """INSERT INTO questions
               (task_id, type, blocks, answer, input_type, response_type,
                position, max_attempts, is_feedback_shown, title, settings)
               VALUES (?, 'subjective', ?, ?, 'text', 'chat',
                       ?, NULL, 1, ?, '{}')""",
            (task_id, blocks_json, answer_json, i, q["title"]),
        )
        print(f"  Q{i+1}: {q['title']} (id={c.lastrowid})")

    conn.commit()
    conn.close()
    print(f"\nDone! Seeded {len(QUESTIONS)} questions into task {task_id}")
    print(f"View at: http://localhost:3000/school/code404?course_id={course_id}&cohort_id=1&view=mentor")


if __name__ == "__main__":
    seed()
