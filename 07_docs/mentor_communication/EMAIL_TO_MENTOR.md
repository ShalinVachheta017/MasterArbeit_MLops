# Email to Mentor - Short Version

**Subject:** Request for Training Labels + Thesis Registration Guidance

---

Dear [Mentor's Name],

I hope this email finds you well.

I have completed Phase 1 of my thesis (data preprocessing and model analysis - 25% complete) and need your guidance to proceed with the next phases.

### Quick Summary

✅ **Completed:**
- Built preprocessing pipeline - processed March 2025 sensor data (181,699 samples at 50Hz)
- Analyzed 1D-CNN-BiLSTM model - confirmed it expects 200 timesteps × 6 sensors → 11 classes
- Verified data quality - 95.1% sensor alignment, only 0.014% missing values

🔴 **Critical Issue Found:**
- **No training labels** found in any data files (preprocessed outputs, original f_data_50hz.csv, or raw Excel files)
- All files contain only sensor readings (Ax, Ay, Az, Gx, Gy, Gz) but no anxiety level classifications
- Cannot proceed with training pipeline without labels

### What I Need

**1. Training Labels**
- Where can I find the labeled data for the March 2025 sensor samples?
- What do the 11 classes represent (anxiety levels 0-10, mental states, etc.)?
- How should I match labels to timestamps?

**2. Training Details** (for reproducibility)
- Hyperparameters you used (batch size, epochs, normalization method)
- How you created the 200-timestep windows (overlap, padding, etc.)
- Train/validation/test split percentages

**3. Model Performance**
- What accuracy/F1-score did the model achieve?
- Any known issues or class imbalances?

**4. Alternative Path**
- If labels unavailable, can I focus thesis purely on MLOps infrastructure (deployment, monitoring) using your pre-trained model for inference only?

### Additional Request - Thesis Registration

**My thesis will be registered on November 1st, 2025.** I need guidance on completing the registration form:
- Should I include training pipeline in scope, or just MLOps infrastructure (depends on label availability)?
- Expected deliverables and timeline to specify?
- Which sections need your signature/approval?

I'm attaching the registration form and will include it with this email.

### Attachments

1. **MENTOR_REQUEST_DETAILED.md** - Complete details of:
   - All work completed (preprocessing, model analysis, data quality)
   - Exact files analyzed and columns found
   - 7 detailed questions with explanations of why I need each piece of information
   - What is blocked and what I'll do immediately after receiving labels
   - Timeline impact analysis

2. **Thesis Registration Form** - To be completed with your guidance

### Timeline

- **If labels received this week:** Thesis stays on track (6-month timeline)
- **If labels delayed 2-3 weeks:** Tight but manageable
- **If labels unavailable:** Need to pivot to MLOps-only approach (update registration form accordingly)

### Request

Please review the attached detailed document when you have time. It contains all context and specific questions. A response this week would be ideal to keep the thesis on schedule for the November 1st registration.

If it's easier to discuss in person, I'm available for a brief meeting at your convenience.

Thank you very much for your guidance and support!

Best regards,  
[Your Name]  
[Student ID]  
[Email]  
[Phone]

---

## 📎 Files to Attach

1. ✅ **MENTOR_REQUEST_DETAILED.md** (this complete breakdown)
2. ✅ **Thesis Registration Form** (your university form as PDF)

---

## ✉️ How to Send

**Option 1: Email with Attachments**
```
To: [Mentor Email]
Subject: Request for Training Labels + Thesis Registration Guidance
Body: Copy the email text above
Attachments: 
  - MENTOR_REQUEST_DETAILED.md
  - [Your Thesis Registration Form].pdf
```

**Option 2: Email with Cloud Link (if attachments too large)**
```
Upload to Google Drive/Dropbox:
  - MENTOR_REQUEST_DETAILED.md
  - Thesis Registration Form

Email body includes link to folder
```

---

**Send this today to stay on schedule!** 📨
