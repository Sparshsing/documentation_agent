"use client";

import React, { useState } from "react";

interface IndexRequestModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface IndexRequestForm {
  index_name: string;
  description: string;
  source_url: string;
  source_type: "website" | "github" | "other" ;
  requester_name: string;
  requester_email: string;
  additional_notes: string;
}

interface FastAPIValidationError {
  loc: (string | number)[];
  msg: string;
  type: string;
  ctx?: {
    expected: string;
  };
}

export default function IndexRequestModal({ isOpen, onClose }: IndexRequestModalProps) {
  const [formData, setFormData] = useState<IndexRequestForm>({
    index_name: "",
    description: "",
    source_url: "",
    source_type: "website",
    requester_name: "",
    requester_email: "",
    additional_notes: "",
  });
  
  const [loading, setLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement | HTMLSelectElement>
  ) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const response = await fetch(`${apiUrl}/api/v1/request-index`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        if (errorData && errorData.detail) {
            if (typeof errorData.detail === 'string') {
                throw new Error(errorData.detail);
            } else if (Array.isArray(errorData.detail)) {
                const messages = errorData.detail.map((d: FastAPIValidationError) => {
                    const field = d.loc && d.loc.length > 1 ? `\`${d.loc[1]}\`` : 'field';
                    let message = d.msg;
                    // Make it more user-friendly
                    if (d.type === 'literal_error' && d.ctx?.expected) {
                        message = `Invalid value for ${field}. Expected one of: ${d.ctx.expected}`;
                    } else {
                        message = `Error in ${field}: ${d.msg}`
                    }
                    return message;
                }).join('; ');
                throw new Error(messages);
            }
        }
        throw new Error(`Request failed with status: ${response.status}`);
      }

      // const result = await response.json();
      setSuccess(true);
      
      // Reset form after successful submission
      setTimeout(() => {
        handleClose();
      }, 2000);
      
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  const handleClose = () => {
    setFormData({
      index_name: "",
      description: "",
      source_url: "",
      source_type: "website",
      requester_name: "",
      requester_email: "",
      additional_notes: "",
    });
    setError(null);
    setSuccess(false);
    setLoading(false);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      {/* Backdrop */}
      <div 
        className="absolute inset-0 bg-black/50 dark:bg-black/70" 
        onClick={handleClose}
      />
      
      {/* Modal */}
      <div className="relative w-full max-w-2xl mx-4 bg-white dark:bg-neutral-900 rounded-lg shadow-xl border border-neutral-200 dark:border-neutral-700 max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-neutral-200 dark:border-neutral-700">
          <h2 className="text-xl font-semibold text-neutral-900 dark:text-neutral-100">
            Request New Index
          </h2>
          <button
            onClick={handleClose}
            className="text-neutral-400 hover:text-neutral-600 dark:hover:text-neutral-300 text-2xl leading-none"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          {success ? (
            <div className="text-center py-8">
              <div className="w-16 h-16 mx-auto mb-4 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
                <svg className="w-8 h-8 text-green-600 dark:text-green-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
              </div>
              <h3 className="text-lg font-semibold text-neutral-900 dark:text-neutral-100 mb-2">
                Request Submitted!
              </h3>
              <p className="text-neutral-600 dark:text-neutral-400">
                Your index creation request has been submitted successfully. We&apos;ll review it and get back to you soon.
              </p>
            </div>
          ) : (
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Index Name */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Index Name *
                  </label>
                  <input
                    type="text"
                    name="index_name"
                    value={formData.index_name}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                    placeholder="e.g., pytorch-docs"
                    required
                  />
                </div>

                {/* Source Type */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Source Type *
                  </label>
                  <select
                    name="source_type"
                    value={formData.source_type}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                    required
                  >
                    <option value="website">Website</option>
                    <option value="github">GitHub</option>
                    <option value="other">Other</option>
                  </select>
                </div>
              </div>

              {/* Source URL */}
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Source URL *
                </label>
                <input
                  type="url"
                  name="source_url"
                  value={formData.source_url}
                  onChange={handleInputChange}
                  className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                  placeholder="https://pytorch.org/docs/"
                  required
                />
              </div>

              {/* Description */}
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Description *
                </label>
                <textarea
                  name="description"
                  value={formData.description}
                  onChange={handleInputChange}
                  rows={3}
                  className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                  placeholder="Brief description of what this index will contain..."
                  required
                />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Requester Name */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Your Name
                  </label>
                  <input
                    type="text"
                    name="requester_name"
                    value={formData.requester_name}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                    placeholder="Optional"
                  />
                </div>

                {/* Requester Email */}
                <div>
                  <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                    Your Email
                  </label>
                  <input
                    type="email"
                    name="requester_email"
                    value={formData.requester_email}
                    onChange={handleInputChange}
                    className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                    placeholder="Optional"
                  />
                </div>
              </div>

              {/* Additional Notes */}
              <div>
                <label className="block text-sm font-medium text-neutral-700 dark:text-neutral-300 mb-2">
                  Additional Notes
                </label>
                <textarea
                  name="additional_notes"
                  value={formData.additional_notes}
                  onChange={handleInputChange}
                  rows={2}
                  className="w-full p-3 rounded-md border border-neutral-300 dark:border-neutral-600 bg-white dark:bg-neutral-800 text-neutral-900 dark:text-neutral-100 text-sm"
                  placeholder="Any additional information or requirements..."
                />
              </div>

              {/* Error */}
              {error && (
                <div className="p-3 bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200 rounded-md text-sm">
                  {error}
                </div>
              )}

              {/* Actions */}
              <div className="flex gap-3 justify-end pt-4">
                <button
                  type="button"
                  onClick={handleClose}
                  className="px-4 py-2 text-neutral-600 dark:text-neutral-400 hover:text-neutral-800 dark:hover:text-neutral-200 text-sm"
                  disabled={loading}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={loading}
                  className="bg-black dark:bg-white text-white dark:text-black px-6 py-2 rounded-md hover:opacity-90 disabled:opacity-50 text-sm"
                >
                  {loading ? "Submitting..." : "Submit Request"}
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
} 