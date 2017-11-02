class Mailer
  def self.client
    Postmark::ApiClient.new(SETTINGS["postmark_api_key"])
  end

  def self.send(to, subject, html_body, from=SETTINGS["email_from_address"])
    return nil if to == SETTINGS["email_from_address"]
    client.deliver(
      from: from,
      to: to,
      subject: subject,
      html_body: html_body,
      track_opens: true
    )
  end
end